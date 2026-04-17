import torch

from ..utils.seeds import TorchConfig
from .base import Diffusion


class GBM(Diffusion):
    """Geometric Brownian Motion under the risk-neutral measure.

    Parameters
    ----------
    r : float
        Risk-free rate.
    sigma : float or list[float]
        Volatility per asset.  Scalar → replicated to all d assets.
    q : float or list[float], default 0.0
        Continuous dividend yield per asset.
    rho : Tensor or None, shape (d, d)
        Correlation matrix.  None → identity (independent assets).
    d : int, default 1
        Number of assets.  Inferred from sigma if sigma is a list.
    """

    def __init__(
        self,
        r: float,
        sigma: float | list[float],
        q: float | list[float] = 0.0,
        rho: torch.Tensor | None = None,
        d: int = 1,
    ):
        if isinstance(sigma, (list, tuple)):
            d = len(sigma)
        self.d = d
        self.r = r
        self._sigma = [sigma] * d if isinstance(sigma, (int, float)) else list(sigma)
        self._q = [q] * d if isinstance(q, (int, float)) else list(q)
        assert len(self._sigma) == d
        assert len(self._q) == d

        if rho is not None:
            assert rho.shape == (d, d)
            self._rho = rho
        else:
            self._rho = torch.eye(d)

        self._L = torch.linalg.cholesky(self._rho.double())

    @property
    def state_dim(self) -> int:
        return self.d

    @property
    def full_dim(self) -> int:
        return self.d

    def simulate(
        self,
        S0: torch.Tensor,
        time_grid: torch.Tensor,
        n_paths: int,
        cfg: TorchConfig | None = None,
    ) -> torch.Tensor:
        """Exact log-normal simulation.

        Paths are generated in ``cfg.sim_dtype`` (default float32)
        for speed.  The exact log-normal scheme has no discretisation
        error, so float32 precision is sufficient for simulation.

        Returns
        -------
        paths : Tensor, shape (n_paths, n_steps+1, d), dtype=sim_dtype
        """
        cfg = self._resolve_cfg(S0, cfg)
        sd = cfg.sim_dtype
        d = self.d
        n_steps = len(time_grid) - 1

        sigma = torch.tensor(self._sigma, device=cfg.device, dtype=sd)
        q = torch.tensor(self._q, device=cfg.device, dtype=sd)
        L = self._L.to(device=cfg.device, dtype=sd)
        S0 = S0.to(device=cfg.device, dtype=sd)
        time_grid = time_grid.to(
            device=cfg.device, dtype=torch.float64
        )  # keep dt precise

        if S0.dim() == 0:
            S0 = S0.unsqueeze(0)
        assert S0.shape == (d,), f"S0 must have shape ({d},), got {S0.shape}"

        drift_rate = self.r - q - 0.5 * sigma**2

        paths = cfg.sim_empty(n_paths, n_steps + 1, d)
        paths[:, 0, :] = S0.unsqueeze(0)

        for i in range(n_steps):
            dt = (time_grid[i + 1] - time_grid[i]).item()
            sqrt_dt = dt**0.5

            Z = cfg.sim_randn(n_paths, d)
            W = Z @ L.T

            log_increment = drift_rate * dt + sigma * sqrt_dt * W
            paths[:, i + 1, :] = paths[:, i, :] * torch.exp(log_increment)

        return paths

    def simulate_batch(
        self,
        S0_batch: torch.Tensor,
        time_grid: torch.Tensor,
        cfg: TorchConfig | None = None,
    ) -> torch.Tensor:
        """Simulate one path per initial state in S0_batch.

        Parameters
        ----------
        S0_batch : Tensor, shape (n_batch, d)
            Each row is a distinct initial state.
        time_grid : Tensor, shape (n_steps+1,)
        cfg : TorchConfig or None

        Returns
        -------
        paths : Tensor, shape (n_batch, n_steps+1, d), dtype=sim_dtype
        """
        cfg = self._resolve_cfg(S0_batch, cfg)
        sd = cfg.sim_dtype
        d = self.d
        n_batch = S0_batch.shape[0]
        n_steps = len(time_grid) - 1

        sigma = torch.tensor(self._sigma, device=cfg.device, dtype=sd)
        q = torch.tensor(self._q, device=cfg.device, dtype=sd)
        L = self._L.to(device=cfg.device, dtype=sd)
        S0_batch = S0_batch.to(device=cfg.device, dtype=sd)
        time_grid = time_grid.to(device=cfg.device, dtype=torch.float64)

        drift_rate = self.r - q - 0.5 * sigma**2

        paths = cfg.sim_empty(n_batch, n_steps + 1, d)
        paths[:, 0, :] = S0_batch

        for i in range(n_steps):
            dt = (time_grid[i + 1] - time_grid[i]).item()
            sqrt_dt = dt**0.5
            Z = cfg.sim_randn(n_batch, d)
            W = Z @ L.T
            log_increment = drift_rate * dt + sigma * sqrt_dt * W
            paths[:, i + 1, :] = paths[:, i, :] * torch.exp(log_increment)

        return paths
