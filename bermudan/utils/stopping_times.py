import torch

from ..networks.features import build_features
from ..networks.feedforward import FeedForward
from ..options.bermudan import BermudanOption


@torch.no_grad()
def stopping_times_pg(
    policy: FeedForward,
    option: BermudanOption,
    S0: torch.Tensor,
    n_paths: int,
) -> torch.Tensor:
    """Extract stopping times under a trained PG policy.

    Returns
    -------
    stop_times : Tensor, shape (n_paths,)
        The exercise time for each path (in real time, not index).
    """
    cfg = option.cfg
    policy.eval()

    paths = option.simulate(S0, n_paths)
    obs = option.observable_at_exercise(paths)
    ex_dates = option.exercise_dates
    payoff = option.payoff
    T = option.T
    N_S = option.N_S
    M = n_paths

    stopped = torch.zeros(M, dtype=torch.bool, device=cfg.device)
    stop_times = torch.full(
        (M,), ex_dates[-1].item(), device=cfg.device, dtype=cfg.dtype
    )

    for n in range(N_S):
        alive = ~stopped
        if alive.sum() == 0:
            break

        S_n = obs[alive, n, :]
        phi = build_features(payoff, S_n, ex_dates[n], T).float()
        logit = policy(phi).squeeze(-1)

        if n == N_S - 1:
            do_stop = torch.ones(alive.sum(), dtype=torch.bool, device=cfg.device)
        else:
            do_stop = (logit >= 0) & (payoff(S_n) > 0)

        alive_idx = torch.where(alive)[0]
        stop_idx = alive_idx[do_stop]
        stop_times[stop_idx] = ex_dates[n].item()
        stopped[stop_idx] = True

    policy.train()
    return stop_times


@torch.no_grad()
def stopping_times_dos(
    networks: list[FeedForward | None],
    option: BermudanOption,
    S0: torch.Tensor,
    n_paths: int,
) -> torch.Tensor:
    """Extract stopping times under a trained DOS cascade."""
    cfg = option.cfg
    paths = option.simulate(S0, n_paths)
    obs = option.observable_at_exercise(paths)
    ex_dates = option.exercise_dates
    payoff = option.payoff
    T = option.T
    N_S = option.N_S
    M = n_paths

    stopped = torch.zeros(M, dtype=torch.bool, device=cfg.device)
    stop_times = torch.full(
        (M,), ex_dates[-1].item(), device=cfg.device, dtype=cfg.dtype
    )

    for n in range(N_S):
        alive = ~stopped
        if alive.sum() == 0:
            break

        if n == N_S - 1:
            stop_times[alive] = ex_dates[n].item()
            stopped[alive] = True
        elif networks[n] is not None:
            S_n = obs[alive, n, :]
            phi = build_features(payoff, S_n, ex_dates[n], T).float()
            logit = networks[n](phi).squeeze(-1)
            do_stop = logit >= 0

            alive_idx = torch.where(alive)[0]
            stop_idx = alive_idx[do_stop]
            stop_times[stop_idx] = ex_dates[n].item()
            stopped[stop_idx] = True

    return stop_times


def stopping_times_lsmc(
    option: BermudanOption,
    S0: torch.Tensor,
    n_paths: int,
) -> torch.Tensor:
    """Extract stopping times from LSMC (run it and return stop indices).

    Returns stop_times in real time units.
    """
    from ..methods.lsmc import LSMC

    cfg = option.cfg
    lsmc = LSMC(degree=3, use_payoff_in_basis=True)
    res = lsmc.price(option, S0, n_paths)

    stop_idx = res.info["stop_idx"]  # integer indices into exercise dates
    ex_dates = option.exercise_dates.cpu()
    stop_times = ex_dates[stop_idx]

    return stop_times
