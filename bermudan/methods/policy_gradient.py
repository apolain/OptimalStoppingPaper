import torch
import torch.optim as optim

from ..networks.features import build_features
from ..networks.feedforward import FeedForward
from ..options.bermudan import BermudanOption
from ..utils.timing import timer
from .base import PricingMethod, PricingResult


class PolicyGradient(PricingMethod):
    """REINFORCE with baseline for Bermudan option pricing.

    The policy network outputs pi_theta(stop | phi(S, t, T)) in (0, 1).
    At each exercise date along a forward trajectory, we sample
    a ∼ Bernoulli(pi_theta) until the first stop or maturity.

    Parameters
    ----------
    hidden_dims : list[int]
        Hidden layer widths of the shared policy network.
    activation : str
        Hidden-layer activation ("relu" or "tanh").
    lr : float
        Learning rate for Adam.
    n_epochs : int
        Number of training epochs (full batch per epoch).
    batch_size : int
        Number of Monte Carlo paths per epoch.
    entropy_coeff : float
        Coefficient lambda_H for entropy regularisation.
        H(pi) = -[p log p + (1-p) log(1-p)], encourages exploration.
    clip_grad_norm : float or None
        Maximum gradient norm for clipping.  None → no clipping.
    """

    def __init__(
        self,
        hidden_dims: list[int] | None = None,
        activation: str = "relu",
        lr: float = 1e-3,
        n_epochs: int = 300,
        batch_size: int = 200_000,
        entropy_coeff: float = 0.01,
        clip_grad_norm: float | None = 1.0,
    ):
        self.hidden_dims = hidden_dims or [64, 64]
        self.activation = activation
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.entropy_coeff = entropy_coeff
        self.clip_grad_norm = clip_grad_norm

    def price(
        self,
        option: BermudanOption,
        S0: torch.Tensor,
        n_paths: int,
        device: torch.device | None = None,
        **kwargs,
    ) -> PricingResult:
        """Train a policy and estimate the option price.

        Parameters
        ----------
        n_paths : int
            Number of paths for the *final* evaluation (separate from training).
        """
        dev = device or torch.device("cpu")
        N_S = option.N_S
        r = option.r
        T = option.T
        payoff = option.payoff
        n_feat = payoff.n_features

        # --- Build policy network ---
        policy = FeedForward(
            input_dim=n_feat,
            hidden_dims=self.hidden_dims,
            output_dim=1,
            activation=self.activation,
            output_activation="sigmoid",
        ).to(dev)

        optimizer = optim.Adam(policy.parameters(), lr=self.lr)

        with timer() as t:
            loss_history: list[float] = []
            reward_history: list[float] = []

            # ========================
            #  Training loop
            # ========================
            for epoch in range(self.n_epochs):
                # Simulate fresh batch of paths
                paths = option.simulate(S0, self.batch_size, device=dev)
                obs = option.observable_at_exercise(paths)  # (M, N_S, state_dim)
                ex_dates = option.exercise_dates.to(dev)

                M = self.batch_size

                # Forward roll: sample actions at each exercise date
                log_probs_sum = torch.zeros(M, device=dev)  # sum of log pi(a_k | x_k)
                entropy_sum = torch.zeros(M, device=dev)
                stopped = torch.zeros(M, dtype=torch.bool, device=dev)
                rewards = torch.zeros(M, device=dev)
                stop_times = torch.full((M,), ex_dates[-1].item(), device=dev)

                for n in range(N_S):
                    alive = ~stopped
                    if alive.sum() == 0:
                        break

                    S_n = obs[alive, n, :]
                    t_n = ex_dates[n]
                    phi = build_features(payoff, S_n, t_n, T)

                    prob_stop = policy(phi).squeeze(-1)  # (M_alive,)
                    prob_stop = prob_stop.clamp(1e-8, 1.0 - 1e-8)

                    # At maturity, force exercise
                    if n == N_S - 1:
                        action = torch.ones_like(prob_stop)
                    else:
                        action = torch.bernoulli(prob_stop)

                    # Log-probability of sampled action
                    log_p = action * torch.log(prob_stop) + (1 - action) * torch.log(
                        1 - prob_stop
                    )

                    # Binary entropy
                    ent = -(
                        prob_stop * torch.log(prob_stop)
                        + (1 - prob_stop) * torch.log(1 - prob_stop)
                    )

                    # Accumulate for alive paths
                    alive_idx = torch.where(alive)[0]
                    log_probs_sum[alive_idx] += log_p
                    entropy_sum[alive_idx] += ent

                    # Record stops
                    do_stop = action == 1.0
                    newly_stopped_local = do_stop
                    newly_stopped_global = alive_idx[newly_stopped_local]

                    rewards[newly_stopped_global] = torch.exp(
                        torch.tensor(-r * t_n.item(), device=dev)
                    ) * payoff(obs[newly_stopped_global, n, :])
                    stop_times[newly_stopped_global] = t_n.item()
                    stopped[newly_stopped_global] = True

                # --- REINFORCE loss ---
                baseline = rewards.mean()
                advantage = rewards - baseline

                # Loss = -E[(R - b) * sum log pi] - lambda_H * E[H]
                pg_loss = -(advantage.detach() * log_probs_sum).mean()
                entropy_loss = -self.entropy_coeff * entropy_sum.mean()
                loss = pg_loss + entropy_loss

                optimizer.zero_grad()
                loss.backward()

                if self.clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        policy.parameters(), self.clip_grad_norm
                    )

                optimizer.step()

                loss_history.append(loss.item())
                reward_history.append(rewards.mean().item())

            # ========================
            #  Evaluation (greedy)
            # ========================
            price_val, std_val = self._evaluate(policy, option, S0, n_paths, dev)

        return PricingResult(
            price=price_val,
            std=std_val,
            elapsed=t.elapsed,
            info={
                "policy": policy,
                "loss_history": loss_history,
                "reward_history": reward_history,
                "n_params": policy.n_params,
            },
        )

    @staticmethod
    @torch.no_grad()
    def _evaluate(
        policy: FeedForward,
        option: BermudanOption,
        S0: torch.Tensor,
        n_paths: int,
        device: torch.device,
    ) -> tuple[float, float]:
        """Evaluate learned policy greedily (threshold 0.5) on fresh paths."""
        policy.eval()

        paths = option.simulate(S0, n_paths, device=device)
        obs = option.observable_at_exercise(paths)
        ex_dates = option.exercise_dates.to(device)
        payoff = option.payoff
        r = option.r
        T = option.T
        M = n_paths
        N_S = option.N_S

        stopped = torch.zeros(M, dtype=torch.bool, device=device)
        cashflow = torch.zeros(M, device=device)

        for n in range(N_S):
            alive = ~stopped
            if alive.sum() == 0:
                break

            S_n = obs[alive, n, :]
            t_n = ex_dates[n]
            phi = build_features(payoff, S_n, t_n, T)
            prob = policy(phi).squeeze(-1)

            if n == N_S - 1:
                do_stop = torch.ones(alive.sum(), dtype=torch.bool, device=device)
            else:
                do_stop = (prob >= 0.5) & (payoff(S_n) > 0)

            alive_idx = torch.where(alive)[0]
            stop_idx = alive_idx[do_stop]

            cashflow[stop_idx] = torch.exp(
                torch.tensor(-r * t_n.item(), device=device)
            ) * payoff(obs[stop_idx, n, :])
            stopped[stop_idx] = True

        # Paths that never stopped (shouldn't happen, but safety)
        never = ~stopped
        if never.sum() > 0:
            cashflow[never] = torch.exp(
                torch.tensor(-r * ex_dates[-1].item(), device=device)
            ) * payoff(obs[never, -1, :])

        policy.train()

        price = cashflow.mean().item()
        std = cashflow.std().item() / (M**0.5)
        return price, std
