from itertools import combinations_with_replacement

import torch

from ..options.bermudan import BermudanOption
from ..utils.timing import timer
from .base import PricingMethod, PricingResult


class LSMC(PricingMethod):
    """Longstaff–Schwartz pricing with polynomial basis.

    Parameters
    ----------
    degree : int, default 2
        Maximum total degree of the polynomial basis.
    use_payoff_in_basis : bool, default True
        If True, append g(S) as an extra regressor.
    """

    def __init__(self, degree: int = 2, use_payoff_in_basis: bool = True):
        self.degree = degree
        self.use_payoff_in_basis = use_payoff_in_basis

    def price(
        self,
        option: BermudanOption,
        S0: torch.Tensor,
        n_paths: int,
        **kwargs,
    ) -> PricingResult:
        cfg = option.cfg

        with timer() as t:
            paths = option.simulate(S0, n_paths)  # (M, N+1, full_dim)
            obs = option.observable_at_exercise(paths)  # (M, N_S, state_dim)

            M = n_paths
            N_S = option.N_S
            r = option.r
            ex_dates = option.exercise_dates  # (N_S,)

            # Payoffs at every exercise date
            payoff_matrix = cfg.zeros(M, N_S)
            for n in range(N_S):
                payoff_matrix[:, n] = option.payoff(obs[:, n, :])

            # Backward induction
            stop_idx = torch.full((M,), N_S - 1, device=cfg.device, dtype=torch.long)
            cashflow = payoff_matrix[:, -1].clone()

            for n in range(N_S - 2, 0, -1):
                dt_to_stop = ex_dates[stop_idx] - ex_dates[n]
                continuation = cashflow * torch.exp(-r * dt_to_stop)

                itm = payoff_matrix[:, n] > 0
                if itm.sum() < 2:
                    continue

                S_itm = obs[itm, n, :]
                Y_itm = continuation[itm]

                X = self._build_basis(S_itm, option.payoff, obs[itm, n, :])

                beta = torch.linalg.lstsq(X, Y_itm.unsqueeze(-1)).solution.squeeze(-1)
                C_hat = X @ beta

                exercise = payoff_matrix[itm, n] >= C_hat
                idx_itm = torch.where(itm)[0]
                idx_exercise = idx_itm[exercise]

                cashflow[idx_exercise] = payoff_matrix[idx_exercise, n]
                stop_idx[idx_exercise] = n

            # Price at t=0
            discount_factors = torch.exp(-r * ex_dates[stop_idx])
            discounted = cashflow * discount_factors

            price = discounted.mean().item()
            std = discounted.std().item() / (M**0.5)

        return PricingResult(
            price=price,
            std=std,
            elapsed=t.elapsed,
            info={
                "stop_idx": stop_idx.cpu(),
                "train_time": t.elapsed,
                "eval_time": 0.0,
            },
        )

    def _build_basis(
        self,
        S: torch.Tensor,
        payoff,
        S_raw: torch.Tensor,
    ) -> torch.Tensor:
        """Build polynomial basis matrix.

        Uses raw (un-normalised) asset prices; numerical stability is
        ensured by the dtype (float64 recommended).
        """
        M, d = S.shape
        dev = S.device
        dt = S.dtype

        columns = [torch.ones(M, device=dev, dtype=dt)]
        indices = list(range(d))

        for deg in range(1, self.degree + 1):
            for combo in combinations_with_replacement(indices, deg):
                col = torch.ones(M, device=dev, dtype=dt)
                for idx in combo:
                    col = col * S[:, idx]
                columns.append(col)

        if self.use_payoff_in_basis:
            columns.append(payoff(S_raw))

        return torch.stack(columns, dim=-1)
