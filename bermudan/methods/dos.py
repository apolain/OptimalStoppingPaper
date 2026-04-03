import torch
import torch.nn as nn
import torch.optim as optim

from ..networks.features import build_features
from ..networks.feedforward import FeedForward
from ..options.bermudan import BermudanOption
from ..utils.timing import timer
from .base import PricingMethod, PricingResult


class DOS(PricingMethod):
    """Deep Optimal Stopping.

    At each exercise date n (backward from N_S-2 to 0), a network f_n
    is trained to output a stopping indicator:

        f_n(phi(S, t_n, T)) ∈ (0, 1)

    Training minimises the negative expected discounted payoff, where
    the decision rule at dates > n is already fixed from earlier
    backward steps.

    Parameters
    ----------
    hidden_dims : list[int]
        Hidden layer widths for each per-date network.
    activation : str
        Activation in hidden layers ("relu" or "tanh").
    lr : float
        Learning rate for Adam.
    n_epochs : int
        Training epochs per exercise date.
    batch_size : int
        Mini-batch size for training.  If 0, use full batch.
    """

    def __init__(
        self,
        hidden_dims: list[int] | None = None,
        activation: str = "relu",
        lr: float = 1e-3,
        n_epochs: int = 200,
        batch_size: int = 0,
    ):
        self.hidden_dims = hidden_dims or [64, 64]
        self.activation = activation
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size

    def price(
        self,
        option: BermudanOption,
        S0: torch.Tensor,
        n_paths: int,
        device: torch.device | None = None,
        **kwargs,
    ) -> PricingResult:
        dev = device or torch.device("cpu")

        with timer() as t:
            # --- Simulate paths ---
            paths = option.simulate(S0, n_paths, device=dev)
            obs = option.observable_at_exercise(paths)  # (M, N_S, state_dim)

            M = n_paths
            N_S = option.N_S
            r = option.r
            ex_dates = option.exercise_dates.to(dev)
            T = option.T
            payoff = option.payoff
            n_feat = payoff.n_features

            # Payoffs at each exercise date
            payoff_matrix = torch.zeros(M, N_S, device=dev)
            for n in range(N_S):
                payoff_matrix[:, n] = payoff(obs[:, n, :])

            # Discount factors from each exercise date to t=0
            discount = torch.exp(-r * ex_dates).to(dev)  # (N_S,)

            # Networks: one per exercise date (except last, where we always exercise)
            networks: list[FeedForward | None] = [None] * N_S

            # stopped[m] = True if path m has been stopped at or before current date
            stopped = torch.zeros(M, dtype=torch.bool, device=dev)

            # cashflow[m] = discounted payoff obtained by path m
            cashflow = torch.zeros(M, device=dev)

            # At maturity: always exercise remaining paths
            alive_last = ~stopped
            cashflow[alive_last] = discount[-1] * payoff_matrix[alive_last, -1]
            stopped[alive_last] = True

            # --- Backward training ---
            loss_history: dict[int, list[float]] = {}

            for n in range(N_S - 2, -1, -1):
                # Future value for alive paths: what they'll get if they continue
                # This is already determined by decisions at dates > n
                future_val = cashflow.clone()  # discounted to t=0

                # Build features at date n
                S_n = obs[:, n, :]
                t_n = ex_dates[n]
                phi = build_features(payoff, S_n, t_n, T)  # (M, n_feat)

                # Immediate exercise value (discounted to 0)
                imm_val = discount[n] * payoff_matrix[:, n]  # (M,)

                # Train network to decide stop (1) vs continue (0)
                net = FeedForward(
                    input_dim=n_feat,
                    hidden_dims=self.hidden_dims,
                    output_dim=1,
                    activation=self.activation,
                    output_activation="sigmoid",
                ).to(dev)

                optimizer = optim.Adam(net.parameters(), lr=self.lr)
                bs = self.batch_size if self.batch_size > 0 else M
                losses = []

                for epoch in range(self.n_epochs):
                    perm = torch.randperm(M, device=dev)
                    epoch_loss = 0.0
                    n_batches = 0

                    for start in range(0, M, bs):
                        idx = perm[start : start + bs]
                        phi_b = phi[idx]
                        imm_b = imm_val[idx]
                        fut_b = future_val[idx]

                        prob_stop = net(phi_b).squeeze(-1)  # (B,)

                        # Expected value = p * immediate + (1-p) * future
                        value = prob_stop * imm_b + (1.0 - prob_stop) * fut_b

                        loss = -value.mean()
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        epoch_loss += loss.item()
                        n_batches += 1

                    losses.append(epoch_loss / n_batches)

                loss_history[n] = losses
                networks[n] = net

                # --- Apply trained decision (hard threshold at 0.5) ---
                with torch.no_grad():
                    prob = net(phi).squeeze(-1)
                    do_stop = (prob >= 0.5) & (payoff_matrix[:, n] > 0)

                # Update cashflows: paths that stop here get immediate value
                newly_stopped = do_stop & (
                    ~stopped if n > 0 else torch.ones(M, dtype=torch.bool, device=dev)
                )
                # For n=0, all paths must get a value
                if n == 0:
                    # Paths not yet stopped continue to their future value
                    # Paths that stop here get immediate value
                    cashflow[do_stop] = imm_val[do_stop]
                else:
                    cashflow[newly_stopped] = imm_val[newly_stopped]
                    stopped[newly_stopped] = True

            # --- Final price ---
            price_val = cashflow.mean().item()
            std_val = cashflow.std().item() / (M**0.5)

        total_params = sum(net.n_params for net in networks if net is not None)

        return PricingResult(
            price=price_val,
            std=std_val,
            elapsed=t.elapsed,
            info={
                "networks": networks,
                "loss_history": loss_history,
                "total_params": total_params,
            },
        )
