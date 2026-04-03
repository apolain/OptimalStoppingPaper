import torch

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
        # --- dimension ---
        if isinstance(sigma, (list, tuple)):
            d = len(sigma)
        self.d = d

        self.r = r

        # --- per-asset vectors (stored as plain Python; moved to device in simulate) ---
        self._sigma = [sigma] * d if isinstance(sigma, (int, float)) else list(sigma)
        self._q = [q] * d if isinstance(q, (int, float)) else list(q)

        assert len(self._sigma) == d
        assert len(self._q) == d

        # --- correlation → Cholesky ---
        if rho is not None:
            assert rho.shape == (d, d), f"rho must be ({d},{d}), got {rho.shape}"
            self._rho = rho.float()
        else:
            self._rho = torch.eye(d)

        # Cholesky factor L such that L L^T = rho
        self._L = torch.linalg.cholesky(self._rho)

    # ----- Diffusion interface -----

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
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Exact log-normal simulation.

        Returns
        -------
        paths : Tensor, shape (n_paths, n_steps+1, d)
        """
        dev = device or S0.device
        d = self.d
        n_steps = len(time_grid) - 1

        sigma = torch.tensor(self._sigma, device=dev, dtype=torch.float32)  # (d,)
        q = torch.tensor(self._q, device=dev, dtype=torch.float32)  # (d,)
        L = self._L.to(dev)  # (d, d)
        S0 = S0.to(dev).float()

        if S0.dim() == 0:
            S0 = S0.unsqueeze(0)
        assert S0.shape == (d,), f"S0 must have shape ({d},), got {S0.shape}"

        # Pre-compute drift per unit time: (r - q_i - 0.5 sigma_i^2)
        drift_rate = self.r - q - 0.5 * sigma**2  # (d,)

        # Allocate output
        paths = torch.empty(n_paths, n_steps + 1, d, device=dev)
        paths[:, 0, :] = S0.unsqueeze(0)

        for i in range(n_steps):
            dt = (time_grid[i + 1] - time_grid[i]).item()
            sqrt_dt = dt**0.5

            # Independent normals → correlated via Cholesky
            Z = torch.randn(n_paths, d, device=dev)  # (M, d)
            W = Z @ L.T  # (M, d), correlated increments

            # Log-normal exact step: S_{t+dt} = S_t * exp(drift*dt + sigma*sqrt(dt)*W)
            log_increment = drift_rate * dt + sigma * sqrt_dt * W  # (M, d)
            paths[:, i + 1, :] = paths[:, i, :] * torch.exp(log_increment)

        return paths
