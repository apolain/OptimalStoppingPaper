"""Heston stochastic volatility model."""

import torch

from .base import Diffusion


class Heston(Diffusion):
    """Heston model under the risk-neutral measure.

    Simulation uses an Euler–Maruyama scheme with full truncation
    (nu is floored at 0 inside the drift and diffusion).

    The state vector is (S, nu) of dimension 2, but only S (index 0)
    is visible to the payoff — override via `observable`.

    Parameters
    ----------
    r     : float – risk-free rate
    q     : float – dividend yield
    kappa : float – mean-reversion speed
    theta : float – long-run variance
    xi    : float – vol-of-vol
    rho   : float – correlation between S and nu  (scalar in [-1,1])
    """

    def __init__(
        self,
        r: float,
        q: float,
        kappa: float,
        theta: float,
        xi: float,
        rho: float,
    ):
        self.r = r
        self.q = q
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho

    # ----- Diffusion interface -----

    @property
    def state_dim(self) -> int:
        return 1  # only S is payoff-relevant

    @property
    def full_dim(self) -> int:
        return 2  # (S, nu)

    def simulate(
        self,
        S0: torch.Tensor,
        time_grid: torch.Tensor,
        n_paths: int,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Euler–Maruyama with full truncation.

        Parameters
        ----------
        S0 : Tensor, shape (2,)
            Initial state (S_0, nu_0).
        time_grid : Tensor, shape (n_steps+1,)
            Must be fine enough between exercise dates to control
            discretisation bias (typically 100–200 steps per year).

        Returns
        -------
        paths : Tensor, shape (n_paths, n_steps+1, 2)
            paths[:, :, 0] = asset price S
            paths[:, :, 1] = variance process nu
        """
        dev = device or S0.device
        S0 = S0.to(dev).float()
        time_grid = time_grid.to(dev).float()

        assert S0.shape == (2,), f"S0 must be (S_0, nu_0), got shape {S0.shape}"

        n_steps = len(time_grid) - 1
        s0, nu0 = S0[0], S0[1]

        # Cholesky for 2×2 correlation: [[1, rho], [rho, 1]]
        rho = self.rho
        L11 = 1.0
        L21 = rho
        L22 = (1.0 - rho**2) ** 0.5

        # Allocate
        S = torch.empty(n_paths, n_steps + 1, device=dev)
        nu = torch.empty(n_paths, n_steps + 1, device=dev)
        S[:, 0] = s0
        nu[:, 0] = nu0

        for i in range(n_steps):
            dt = (time_grid[i + 1] - time_grid[i]).item()
            sqrt_dt = dt**0.5

            nu_pos = nu[:, i].clamp(min=0.0)  # full truncation
            sqrt_nu = nu_pos.sqrt()

            # Correlated Brownian increments
            Z1 = torch.randn(n_paths, device=dev)
            Z2 = torch.randn(n_paths, device=dev)
            dW_S = sqrt_dt * (L11 * Z1)
            dW_nu = sqrt_dt * (L21 * Z1 + L22 * Z2)

            # Variance process (Euler + truncation)
            nu[:, i + 1] = (
                nu[:, i]
                + self.kappa * (self.theta - nu_pos) * dt
                + self.xi * sqrt_nu * dW_nu
            )

            # Asset price (log-Euler for positivity)
            log_inc = (self.r - self.q - 0.5 * nu_pos) * dt + sqrt_nu * dW_S
            S[:, i + 1] = S[:, i] * torch.exp(log_inc)

        # Stack into (n_paths, n_steps+1, 2)
        return torch.stack([S, nu], dim=-1)
