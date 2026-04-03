from itertools import combinations_with_replacement

import numpy as np
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
        In dimension d, the number of basis functions is C(d + degree, degree).
    use_payoff_in_basis : bool, default True
        If True, append g(S) as an extra regressor (improves fit for
        max-call where the payoff structure is non-trivial).
    """

    def __init__(self, degree: int = 2, use_payoff_in_basis: bool = True):
        self.degree = degree
        self.use_payoff_in_basis = use_payoff_in_basis

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
            paths = option.simulate(S0, n_paths, device=dev)  # (M, N+1, full_dim)
            obs = option.observable_at_exercise(paths)  # (M, N_S, state_dim)

            M = n_paths
            N_S = option.N_S
            r = option.r
            ex_dates = option.exercise_dates.to(dev)  # (N_S,)
            d = option.diffusion.state_dim

            # --- Compute payoffs at every exercise date ---
            # payoff_matrix[m, n] = g(S_{u_n}^m)
            payoff_matrix = torch.zeros(M, N_S, device=dev)
            for n in range(N_S):
                payoff_matrix[:, n] = option.payoff(obs[:, n, :])

            # --- Backward induction ---
            # cashflow[m] = discounted cashflow from the current optimal strategy
            # stop_idx[m] = exercise-date index at which path m stops
            stop_idx = torch.full((M,), N_S - 1, device=dev, dtype=torch.long)
            cashflow = payoff_matrix[:, -1].clone()  # at maturity, always exercise

            for n in range(N_S - 2, 0, -1):  # skip n=0 (will handle pricing at t=0)
                dt_to_next = ex_dates[stop_idx] - ex_dates[n]
                continuation = cashflow * torch.exp(-r * dt_to_next)

                # Only regress on paths that are in-the-money
                itm = payoff_matrix[:, n] > 0
                if itm.sum() == 0:
                    continue

                S_itm = obs[itm, n, :]  # (M_itm, d)
                Y_itm = continuation[itm]  # (M_itm,)

                # Build polynomial basis
                X = self._build_basis(S_itm, option.payoff, obs[itm, n, :], n, option)

                # Least-squares regression: Y = X @ beta
                # Use lstsq for numerical stability
                result = torch.linalg.lstsq(X, Y_itm.unsqueeze(-1))
                beta = result.solution.squeeze(-1)  # (n_basis,)

                C_hat = X @ beta  # (M_itm,)

                # Exercise if immediate payoff > estimated continuation
                exercise = payoff_matrix[itm, n] >= C_hat
                idx_itm = torch.where(itm)[0]
                idx_exercise = idx_itm[exercise]

                cashflow[idx_exercise] = payoff_matrix[idx_exercise, n]
                stop_idx[idx_exercise] = n

            # --- Price at t=0 ---
            # Discount each path's cashflow back to t=0
            discount_factors = torch.exp(-r * ex_dates[stop_idx])
            discounted = cashflow * discount_factors

            price = discounted.mean().item()
            std = discounted.std().item() / (M**0.5)

        return PricingResult(
            price=price,
            std=std,
            elapsed=t.elapsed,
            info={"stop_idx": stop_idx.cpu()},
        )

    def _build_basis(
        self,
        S: torch.Tensor,
        payoff,
        S_raw: torch.Tensor,
        n: int,
        option: BermudanOption,
    ) -> torch.Tensor:
        """Build polynomial basis matrix.

        Parameters
        ----------
        S : Tensor, shape (M_itm, d)
        payoff : Payoff
        S_raw : Tensor, shape (M_itm, d)
        n : int – exercise date index
        option : BermudanOption

        Returns
        -------
        X : Tensor, shape (M_itm, n_basis)
        """
        M, d = S.shape
        dev = S.device

        # Normalise inputs for numerical stability
        S_norm = S / (payoff.K + 1e-8)

        # Generate all monomials up to `degree`
        columns = [torch.ones(M, device=dev)]  # constant term
        indices = list(range(d))

        for deg in range(1, self.degree + 1):
            for combo in combinations_with_replacement(indices, deg):
                col = torch.ones(M, device=dev)
                for idx in combo:
                    col = col * S_norm[:, idx]
                columns.append(col)

        if self.use_payoff_in_basis:
            columns.append(payoff(S_raw))

        return torch.stack(
            columns, dim=-1
        )  # (M_itm, n_basis)        return torch.stack(columns, dim=-1)  # (M_itm, n_basis)
