import torch
import torch.nn as nn
import torch.optim as optim

from ..options.bermudan import BermudanOption
from ..utils.timing import timer
from .base import PricingMethod, PricingResult


class DOSNetwork(nn.Module):
    """Network architecture matching Becker et al. exactly.

    BN(input) → [Linear(no bias) → BN → ReLU] × L → Linear → scalar logit

    Parameters
    ----------
    input_dim : int
        Raw state dimension (d for max-call, 1 for put).
    hidden_dims : list[int]
        Widths of hidden layers (paper uses [d+50, d+50]).
    """

    def __init__(self, input_dim: int, hidden_dims: list[int]):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(input_dim)

        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h, bias=False))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            in_dim = h
        # Final layer: scalar logit, no activation
        layers.append(nn.Linear(in_dim, 1))

        self.layers = nn.Sequential(*layers)

        # Xavier uniform init (matching TF default)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self._n_params = sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, input_dim), float32

        Returns
        -------
        logit : (batch, 1)
        """
        if x.shape[0] > 1:
            x = self.input_bn(x)
        return self.layers(x)

    @property
    def n_params(self) -> int:
        return self._n_params


class DOS(PricingMethod):
    """Deep Optimal Stopping.

    Parameters
    ----------
    hidden_dims : list[int] or None
        Hidden layer widths.  None → [d+50, d+50] (paper default).
    lr : float
        Adam learning rate.
    n_iters : int
        Number of gradient steps per exercise date.
    batch_size : int
        Paths per training batch (fresh simulation each iteration).
    n_eval : int
        Paths for final price evaluation (separate from training).
    """

    def __init__(
        self,
        hidden_dims: list[int] | None = None,
        lr: float = 0.001,
        n_iters: int = 1000,
        batch_size: int = 8192,
    ):
        self._hidden_dims_override = hidden_dims
        self.lr = lr
        self.n_iters = n_iters
        self.batch_size = batch_size

    def _get_hidden_dims(self, d: int) -> list[int]:
        if self._hidden_dims_override is not None:
            return self._hidden_dims_override
        return [d + 50, d + 50]

    def price(
        self,
        option: BermudanOption,
        S0: torch.Tensor,
        n_paths: int,
        **kwargs,
    ) -> PricingResult:
        """Train and evaluate. Pass ``logger=ExperimentLogger(...)`` to log."""
        logger = kwargs.get("logger", None)
        cfg = option.cfg

        N_S = option.N_S
        r = option.r
        ex_dates = option.exercise_dates
        T = option.T
        payoff = option.payoff
        d = option.diffusion.state_dim

        hidden_dims = self._get_hidden_dims(d)
        discount = torch.exp(-r * ex_dates).to(cfg.device)

        networks: list[DOSNetwork | None] = [None] * N_S
        loss_history: dict[int, list[float]] = {}

        with timer() as train_timer:
            # --- Backward training: one network per date ---
            for n in range(N_S - 2, -1, -1):
                net = DOSNetwork(
                    input_dim=d,
                    hidden_dims=hidden_dims,
                ).to(cfg.device)

                optimizer = optim.Adam(net.parameters(), lr=self.lr)
                losses = []

                for it in range(self.n_iters):
                    # Fresh batch at each iteration (key to DOS)
                    paths = option.simulate(S0, self.batch_size)
                    obs = option.observable_at_exercise(paths)

                    # Raw state at date n (float32 for network)
                    X_n = obs[:, n, :].float()

                    # Immediate exercise value (discounted to 0)
                    imm = (discount[n] * payoff(obs[:, n, :])).float()

                    # Continuation value from future strategy (hard decisions)
                    with torch.no_grad():
                        # Start with terminal payoff
                        cont = (discount[-1] * payoff(obs[:, -1, :])).float()
                        # Apply each future network backward
                        for k in range(N_S - 2, n, -1):
                            if networks[k] is not None:
                                networks[k].eval()
                                X_k = obs[:, k, :].float()
                                logit_k = networks[k](X_k).squeeze(-1)
                                stop_k = logit_k >= 0
                                imm_k = (discount[k] * payoff(obs[:, k, :])).float()
                                cont = torch.where(stop_k, imm_k, cont)

                    # Train: maximise E[F(X_n)*imm + (1-F(X_n))*cont]
                    net.train()
                    logit = net(X_n).squeeze(-1)
                    F = torch.sigmoid(logit)

                    value = F * imm + (1.0 - F) * cont
                    loss = -value.mean()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    losses.append(loss.item())

                    if logger is not None:
                        logger.log_epoch(date=n, iteration=it, loss=loss.item())

                loss_history[n] = losses
                networks[n] = net

        # --- Evaluation on fresh paths ---
        with timer() as eval_timer:
            price_val, std_val = self._evaluate(networks, option, S0, n_paths)

        total_params = sum(net.n_params for net in networks if net is not None)

        if logger is not None:
            models = {
                f"net_date{n}": net.state_dict()
                for n, net in enumerate(networks)
                if net is not None
            }
            logger.save_models_dict(models)

        return PricingResult(
            price=price_val,
            std=std_val,
            elapsed=train_timer.elapsed,
            info={
                "networks": networks,
                "loss_history": loss_history,
                "total_params": total_params,
                "train_time": train_timer.elapsed,
                "eval_time": eval_timer.elapsed,
            },
        )

    @staticmethod
    @torch.no_grad()
    def _evaluate(
        networks: list[DOSNetwork | None],
        option: BermudanOption,
        S0: torch.Tensor,
        n_paths: int,
    ) -> tuple[float, float]:
        """Evaluate the learned strategy on fresh paths (forward pass)."""
        cfg = option.cfg
        paths = option.simulate(S0, n_paths)
        obs = option.observable_at_exercise(paths)
        ex_dates = option.exercise_dates
        payoff = option.payoff
        r = option.r
        N_S = option.N_S
        M = n_paths
        discount = torch.exp(-r * ex_dates).to(cfg.device)

        # Set all networks to eval mode (use BN running stats)
        for net in networks:
            if net is not None:
                net.eval()

        stopped = torch.zeros(M, dtype=torch.bool, device=cfg.device)
        cashflow = cfg.zeros(M)

        for n in range(N_S):
            alive = ~stopped
            if alive.sum() == 0:
                break

            if n == N_S - 1:
                cashflow[alive] = discount[n] * payoff(obs[alive, n, :])
                stopped[alive] = True
            elif networks[n] is not None:
                X_n = obs[alive, n, :].float()
                logit = networks[n](X_n).squeeze(-1)
                do_stop = logit >= 0

                alive_idx = torch.where(alive)[0]
                stop_idx = alive_idx[do_stop]

                cashflow[stop_idx] = discount[n] * payoff(obs[stop_idx, n, :])
                stopped[stop_idx] = True

        never = ~stopped
        if never.sum() > 0:
            cashflow[never] = discount[-1] * payoff(obs[never, -1, :])

        return cashflow.mean().item(), cashflow.std().item() / (M**0.5)
