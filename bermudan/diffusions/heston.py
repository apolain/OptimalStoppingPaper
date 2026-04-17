import torch

from ..utils.seeds import TorchConfig
from .base import Diffusion


class Heston(Diffusion):
    """Heston model under the risk-neutral measure.

    Simulation uses an Euler–Maruyama scheme with full truncation
    (nu is floored at 0 inside the drift and diffusion).

    The state vector is (S, nu) of dimension 2, but only S (index 0)
    is visible to the payoff — see ``observable``.

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

    @property
    def state_dim(self) -> int:
        return 1

    @property
    def full_dim(self) -> int:
        return 2

    def simulate(
        self,
        S0: torch.Tensor,
        time_grid: torch.Tensor,
        n_paths: int,
        cfg: TorchConfig | None = None,
    ) -> torch.Tensor:
        """Euler–Maruyama with full truncation.

        Paths are generated in ``cfg.sim_dtype`` (default float32).

        Parameters
        ----------
        S0 : Tensor, shape (2,)
            Initial state (S_0, nu_0).
        time_grid : Tensor, shape (n_steps+1,)
        cfg : TorchConfig or None

        Returns
        -------
        paths : Tensor, shape (n_paths, n_steps+1, 2), dtype=sim_dtype
        """
        cfg = self._resolve_cfg(S0, cfg)
        sd = cfg.sim_dtype
        S0 = S0.to(device=cfg.device, dtype=sd)
        time_grid = time_grid.to(device=cfg.device, dtype=torch.float64)

        assert S0.shape == (2,), f"S0 must be (S_0, nu_0), got shape {S0.shape}"

        n_steps = len(time_grid) - 1
        s0, nu0 = S0[0], S0[1]

        rho = self.rho
        L22 = (1.0 - rho**2) ** 0.5

        S = cfg.sim_empty(n_paths, n_steps + 1)
        nu = cfg.sim_empty(n_paths, n_steps + 1)
        S[:, 0] = s0
        nu[:, 0] = nu0

        for i in range(n_steps):
            dt = (time_grid[i + 1] - time_grid[i]).item()
            sqrt_dt = dt**0.5

            nu_pos = nu[:, i].clamp(min=0.0)
            sqrt_nu = nu_pos.sqrt()

            Z1 = cfg.sim_randn(n_paths)
            Z2 = cfg.sim_randn(n_paths)
            dW_S = sqrt_dt * Z1
            dW_nu = sqrt_dt * (rho * Z1 + L22 * Z2)

            nu[:, i + 1] = (
                nu[:, i]
                + self.kappa * (self.theta - nu_pos) * dt
                + self.xi * sqrt_nu * dW_nu
            )

            log_inc = (self.r - self.q - 0.5 * nu_pos) * dt + sqrt_nu * dW_S
            S[:, i + 1] = S[:, i] * torch.exp(log_inc)

        return torch.stack([S, nu], dim=-1)

    def simulate_batch(
        self,
        S0_batch: torch.Tensor,
        time_grid: torch.Tensor,
        cfg: TorchConfig | None = None,
    ) -> torch.Tensor:
        """Simulate one path per initial state in S0_batch.

        Parameters
        ----------
        S0_batch : Tensor, shape (n_batch, 2)
            Each row is (S_0, nu_0).
        time_grid : Tensor, shape (n_steps+1,)

        Returns
        -------
        paths : Tensor, shape (n_batch, n_steps+1, 2), dtype=sim_dtype
        """
        cfg = self._resolve_cfg(S0_batch, cfg)
        sd = cfg.sim_dtype
        S0_batch = S0_batch.to(device=cfg.device, dtype=sd)
        time_grid = time_grid.to(device=cfg.device, dtype=torch.float64)

        n_batch = S0_batch.shape[0]
        n_steps = len(time_grid) - 1

        rho = self.rho
        L22 = (1.0 - rho**2) ** 0.5

        S = cfg.sim_empty(n_batch, n_steps + 1)
        nu = cfg.sim_empty(n_batch, n_steps + 1)
        S[:, 0] = S0_batch[:, 0]
        nu[:, 0] = S0_batch[:, 1]

        for i in range(n_steps):
            dt = (time_grid[i + 1] - time_grid[i]).item()
            sqrt_dt = dt**0.5

            nu_pos = nu[:, i].clamp(min=0.0)
            sqrt_nu = nu_pos.sqrt()

            Z1 = cfg.sim_randn(n_batch)
            Z2 = cfg.sim_randn(n_batch)
            dW_S = sqrt_dt * Z1
            dW_nu = sqrt_dt * (rho * Z1 + L22 * Z2)

            nu[:, i + 1] = (
                nu[:, i]
                + self.kappa * (self.theta - nu_pos) * dt
                + self.xi * sqrt_nu * dW_nu
            )
            log_inc = (self.r - self.q - 0.5 * nu_pos) * dt + sqrt_nu * dW_S
            S[:, i + 1] = S[:, i] * torch.exp(log_inc)

        return torch.stack([S, nu], dim=-1)
