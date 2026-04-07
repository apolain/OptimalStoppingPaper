from dataclasses import dataclass, field

import torch

from ..diffusions.base import Diffusion
from ..payoffs.base import Payoff
from ..utils.seeds import TorchConfig


@dataclass
class BermudanOption:
    """Complete specification of a Bermudan option pricing problem.

    This object is consumed by every pricing method (LSMC, DOS, PG).
    It bundles the model, payoff, contract dates, **and** the numerical
    dtype/device so that all downstream computations are type-consistent.

    Parameters
    ----------
    diffusion : Diffusion
        Risk-neutral dynamics of the underlying(s).
    payoff : Payoff
        Payoff function g(S).
    r : float
        Constant risk-free discount rate.
    T : float
        Maturity in years.
    N : int
        Number of time steps in the simulation grid.
    exercise_indices : list[int] or None
        Indices into the simulation grid that are admissible exercise dates.
        If None, every grid point is an exercise date (N_S = N+1).
    cfg : TorchConfig or None
        Global device + dtype.  If None, defaults to CPU / float64.
    """

    diffusion: Diffusion
    payoff: Payoff
    r: float
    T: float
    N: int
    exercise_indices: list[int] | None = None
    cfg: TorchConfig | None = None

    def __post_init__(self):
        if self.cfg is None:
            self.cfg = TorchConfig.make(device="cpu", dtype=torch.float64)

        self.dt = self.T / self.N
        self.time_grid = self.cfg.linspace(0.0, self.T, self.N + 1)

        if self.exercise_indices is None:
            self.exercise_indices = list(range(self.N + 1))

        self.exercise_dates = self.time_grid[self.exercise_indices]
        self.N_S = len(self.exercise_indices)

    def simulate(
        self,
        S0: torch.Tensor,
        n_paths: int,
    ) -> torch.Tensor:
        """Simulate full paths on the time grid.

        Returns
        -------
        paths : Tensor, shape (n_paths, N+1, full_dim), dtype=sim_dtype
            Paths in simulation precision (float32 by default).
        """
        return self.diffusion.simulate(S0, self.time_grid, n_paths, cfg=self.cfg)

    def observable_at_exercise(
        self,
        paths: torch.Tensor,
    ) -> torch.Tensor:
        """Extract observable states at exercise dates, promoted to cfg.dtype.

        Paths are simulated in sim_dtype (float32) for speed.  This method
        extracts only the exercise-date slices and promotes them to
        cfg.dtype (float64) for downstream numerical operations (regression,
        discounting, payoff comparisons).

        Parameters
        ----------
        paths : Tensor, shape (n_paths, N+1, full_dim), any dtype

        Returns
        -------
        obs : Tensor, shape (n_paths, N_S, state_dim), dtype=cfg.dtype
        """
        full_at_ex = paths[:, self.exercise_indices, :]
        obs = self.diffusion.observable(full_at_ex)
        return obs.to(dtype=self.cfg.dtype)

    def discount(self, t_from: float, t_to: float) -> float:
        """Deterministic discount factor exp(-r * (t_to - t_from))."""
        return torch.exp(torch.tensor(-self.r * (t_to - t_from))).item()
