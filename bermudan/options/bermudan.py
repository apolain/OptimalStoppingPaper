from dataclasses import dataclass

import torch

from ..diffusions.base import Diffusion
from ..payoffs.base import Payoff


@dataclass
class BermudanOption:
    """Complete specification of a Bermudan option pricing problem.

    This object is consumed by every pricing method (LSMC, DOS, PG).
    It bundles the model, payoff, and contract dates so that all methods
    operate on exactly the same problem.

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
        Index 0 corresponds to t=0 and index N to t=T.

    Derived attributes
    ------------------
    time_grid : Tensor, shape (N+1,)
        Uniform grid [0, dt, 2dt, ..., T].
    exercise_dates : Tensor
        Time values of exercise opportunities.
    N_S : int
        Number of exercise dates.
    """

    diffusion: Diffusion
    payoff: Payoff
    r: float
    T: float
    N: int
    exercise_indices: list[int] | None = None

    def __post_init__(self):
        dt = self.T / self.N
        self.dt = dt
        self.time_grid = torch.linspace(0.0, self.T, self.N + 1)

        if self.exercise_indices is None:
            self.exercise_indices = list(range(self.N + 1))

        self.exercise_dates = self.time_grid[self.exercise_indices]
        self.N_S = len(self.exercise_indices)

    def simulate(
        self,
        S0: torch.Tensor,
        n_paths: int,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Simulate full paths on the time grid.

        Returns
        -------
        paths : Tensor, shape (n_paths, N+1, full_dim)
        """
        dev = device or torch.device("cpu")
        grid = self.time_grid.to(dev)
        return self.diffusion.simulate(S0, grid, n_paths, device=dev)

    def observable_at_exercise(
        self,
        paths: torch.Tensor,
    ) -> torch.Tensor:
        """Extract observable states at exercise dates only.

        Parameters
        ----------
        paths : Tensor, shape (n_paths, N+1, full_dim)

        Returns
        -------
        obs : Tensor, shape (n_paths, N_S, state_dim)
        """
        full_at_ex = paths[:, self.exercise_indices, :]  # (M, N_S, full_dim)
        return self.diffusion.observable(full_at_ex)  # (M, N_S, state_dim)

    def discount(self, t_from: float, t_to: float) -> float:
        """Deterministic discount factor exp(-r * (t_to - t_from))."""
        return torch.exp(torch.tensor(-self.r * (t_to - t_from))).item()
