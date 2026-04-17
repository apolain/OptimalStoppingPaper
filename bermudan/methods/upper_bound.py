from dataclasses import dataclass

import torch

from ..networks.features import build_features
from ..networks.feedforward import FeedForward
from ..options.bermudan import BermudanOption
from ..utils.timing import timer


@dataclass
class DualBoundResult:
    lower: float = 0.0
    lower_std: float = 0.0
    upper: float = 0.0
    upper_std: float = 0.0
    gap: float = 0.0
    gap_relative: float = 0.0
    elapsed: float = 0.0


def _discount0(cfg, r: float, t: float) -> torch.Tensor:
    return torch.exp(cfg.tensor(-r * t))


def _policy_stop_decision(
    policy: FeedForward,
    option: BermudanOption,
    S_n_obs: torch.Tensor,
    n: int,
) -> torch.Tensor:
    """Return policy exercise decision at exercise index n for observed states.

    Output shape: (batch,), dtype=bool
    """
    cfg = option.cfg
    payoff = option.payoff
    t_n = option.exercise_dates[n]
    phi = build_features(payoff, S_n_obs, t_n, option.T).float()
    logit = policy(phi).squeeze(-1)

    if n == option.N_S - 1:
        return torch.ones(S_n_obs.shape[0], dtype=torch.bool, device=cfg.device)

    return (logit >= 0) & (payoff(S_n_obs) > 0)


def _greedy_cashflow(
    policy: FeedForward,
    obs: torch.Tensor,
    option: BermudanOption,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stopping date and discounted-to-0 payoff under the fixed greedy policy."""
    cfg = option.cfg
    payoff = option.payoff
    ex_dates = option.exercise_dates
    M = obs.shape[0]
    N_S = option.N_S
    r = option.r

    stopped = torch.zeros(M, dtype=torch.bool, device=cfg.device)
    stop_n = torch.full((M,), N_S - 1, dtype=torch.long, device=cfg.device)
    cashflow0 = cfg.zeros(M)

    for n in range(N_S):
        alive = ~stopped
        if not alive.any():
            break

        S_n = obs[alive, n, :]
        do_stop = _policy_stop_decision(policy, option, S_n, n)

        alive_idx = torch.where(alive)[0]
        idx = alive_idx[do_stop]
        if idx.numel() > 0:
            disc0 = _discount0(cfg, r, ex_dates[n].item())
            cashflow0[idx] = disc0 * payoff(obs[idx, n, :])
            stop_n[idx] = n
            stopped[idx] = True

    # Safety fallback
    never = ~stopped
    if never.any():
        disc0 = _discount0(cfg, r, ex_dates[-1].item())
        cashflow0[never] = disc0 * payoff(obs[never, -1, :])
        stop_n[never] = N_S - 1

    return stop_n, cashflow0


def _rollout_from_state_discounted_to_tn(
    policy: FeedForward,
    option: BermudanOption,
    full_states_at_n: torch.Tensor,
    n: int,
    M_inner: int,
) -> torch.Tensor:
    """Estimate L_n / B_n for each outer state at time n.

    For each outer state S_n, simulate M_inner subpaths from t_n onward,
    apply the fixed policy from date n onward, and return:
        E_n[ g_{tau_n} / B_{tau_n} ]
    i.e. discounted to time 0, not relative to t_n.

    Parameters
    ----------
    full_states_at_n : (M_outer, full_dim)
        Full simulator state at exercise date n.
    n : int
        Exercise index.
    M_inner : int
        Number of inner paths per outer state.

    Returns
    -------
    disc_L_n : (M_outer,)
        Estimate of L_n / B_n for each outer state.
    """
    cfg = option.cfg
    payoff = option.payoff
    ex_dates = option.exercise_dates
    ex_grid_idx = option.exercise_indices
    M_outer = full_states_at_n.shape[0]
    N_S = option.N_S
    r = option.r

    # Build time grid
    sim_start = ex_grid_idx[n]
    sub_grid = option.time_grid[sim_start:]
    remaining = list(range(n, N_S))
    sub_ex_local = [ex_grid_idx[i] - sim_start for i in remaining]

    # Expand outer states into inner batch
    S_exp = full_states_at_n.repeat_interleave(M_inner, dim=0)
    batch = S_exp.shape[0]

    sub_paths = option.diffusion.simulate_batch(S_exp, sub_grid, cfg=cfg)
    obs_sub = option.diffusion.observable(sub_paths[:, sub_ex_local, :]).to(
        dtype=cfg.dtype
    )

    stopped = torch.zeros(batch, dtype=torch.bool, device=cfg.device)
    cf0 = cfg.zeros(batch)

    for k_local, k_global in enumerate(remaining):
        alive = ~stopped
        if not alive.any():
            break

        S_k = obs_sub[alive, k_local, :]
        do_stop = _policy_stop_decision(policy, option, S_k, k_global)

        alive_idx = torch.where(alive)[0]
        idx = alive_idx[do_stop]
        if idx.numel() > 0:
            disc0 = _discount0(cfg, r, ex_dates[k_global].item())
            cf0[idx] = disc0 * payoff(obs_sub[idx, k_local, :])
            stopped[idx] = True

    # Safety fallback
    never = ~stopped
    if never.any():
        disc0 = _discount0(cfg, r, ex_dates[-1].item())
        cf0[never] = disc0 * payoff(obs_sub[never, -1, :])

    return cf0.view(M_outer, M_inner).mean(dim=1)


@torch.no_grad()
def andersen_broadie(
    policy: FeedForward,
    option: BermudanOption,
    S0: torch.Tensor,
    M_outer: int = 5_000,
    M_inner: int = 500,
    inner_batch_size: int | None = None,
) -> DualBoundResult:
    """Compute Andersen-Broadie primal-dual bounds.

    Lower bound:
        independent Monte Carlo estimate of value under the fixed greedy policy.

    Upper bound:
        separate outer paths + nested simulation to estimate the lower-bound
        process L_n/B_n and the conditional expectation terms needed in the
        Andersen-Broadie martingale.
    """
    cfg = option.cfg
    payoff = option.payoff
    ex_dates = option.exercise_dates
    ex_idx = option.exercise_indices
    N_S = option.N_S
    r = option.r
    chunk = inner_batch_size or M_outer

    policy.eval()

    with timer() as clock:
        # 1) Lower bound on independent paths
        paths_lower = option.simulate(S0, M_outer)
        obs_lower = option.observable_at_exercise(paths_lower)
        _, lower_cf0 = _greedy_cashflow(policy, obs_lower, option)

        del paths_lower, obs_lower
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 2) Upper bound on separate outer paths
        paths = option.simulate(S0, M_outer)
        obs = option.observable_at_exercise(paths)

        # discounted exercise values
        disc_g = cfg.zeros(M_outer, N_S)
        for n in range(N_S):
            disc0 = _discount0(cfg, r, ex_dates[n].item())
            disc_g[:, n] = disc0 * payoff(obs[:, n, :])

        # policy indicator
        l = torch.zeros(M_outer, N_S, dtype=torch.bool, device=cfg.device)
        for n in range(N_S):
            l[:, n] = _policy_stop_decision(policy, option, obs[:, n, :], n)

        # discounted lower-bound process estimated by nested MC
        disc_L = cfg.zeros(M_outer, N_S)

        for n in range(N_S):
            if n == N_S - 1:
                disc_L[:, n] = disc_g[:, n]
                print(f"    date {n+1}/{N_S} done (terminal)")
                continue

            ex_now = l[:, n]
            cont_now = ~ex_now

            if ex_now.any():
                disc_L[ex_now, n] = disc_g[ex_now, n]

            if cont_now.any():
                idx_outer = torch.where(cont_now)[0]
                C_vals = cfg.zeros(idx_outer.numel())

                for c0 in range(0, idx_outer.numel(), chunk):
                    c1 = min(c0 + chunk, idx_outer.numel())
                    idx_chunk = idx_outer[c0:c1]
                    full_state = paths[idx_chunk, ex_idx[n], :].to(dtype=cfg.sim_dtype)
                    C_vals[c0:c1] = _rollout_from_state_discounted_to_tn(
                        policy=policy,
                        option=option,
                        full_states_at_n=full_state,
                        n=n,
                        M_inner=M_inner,
                    )

                disc_L[idx_outer, n] = C_vals

            print(f"    date {n+1}/{N_S} done")

        # 3) Andersen-Broadie martingale
        lam = cfg.zeros(M_outer, N_S)
        lam[:, 0] = disc_L[:, 0]

        for k in range(1, N_S):
            prev_ex = l[:, k - 1]  # l_{k-1}
            cond_exp_increment = cfg.zeros(M_outer)

            # Only needed on the exercise region at k-1
            if prev_ex.any():
                idx_outer = torch.where(prev_ex)[0]
                E_disc_Lk = cfg.zeros(idx_outer.numel())

                for c0 in range(0, idx_outer.numel(), chunk):
                    c1 = min(c0 + chunk, idx_outer.numel())
                    idx_chunk = idx_outer[c0:c1]
                    full_state = paths[idx_chunk, ex_idx[k - 1], :].to(
                        dtype=cfg.sim_dtype
                    )

                    E_disc_Lk[c0:c1] = _rollout_from_state_discounted_to_tn(
                        policy=policy,
                        option=option,
                        full_states_at_n=full_state,
                        n=k,
                        M_inner=M_inner,
                    )

                cond_exp_increment[idx_outer] = E_disc_Lk - disc_L[idx_outer, k - 1]

            lam[:, k] = (
                lam[:, k - 1] + disc_L[:, k] - disc_L[:, k - 1] - cond_exp_increment
            )

        # Pathwise AB upper estimator: L0 + max_n(g_n/B_n - lambda_n)
        pathwise_upper = disc_L[:, 0] + (disc_g - lam).max(dim=1).values

    L = lower_cf0.mean().item()
    L_se = lower_cf0.std(unbiased=True).item() / (M_outer**0.5)
    U = pathwise_upper.mean().item()
    U_se = pathwise_upper.std(unbiased=True).item() / (M_outer**0.5)

    policy.train()

    return DualBoundResult(
        lower=L,
        lower_std=L_se,
        upper=U,
        upper_std=U_se,
        gap=U - L,
        gap_relative=(U - L) / max(abs(L), 1e-10),
        elapsed=clock.elapsed,
    )
