from abc import ABC, abstractmethod

import torch


class Diffusion(ABC):
    """Base class for simulatable diffusion processes.

    Convention
    ----------
    All diffusions operate under the risk-neutral measure Q.
    The `simulate` method returns paths on a user-supplied time grid,
    which may be coarser or finer than the exercise dates.

    Subclasses must implement `simulate`.  They may optionally override
    `state_dim` (number of state variables visible to the payoff) and
    `full_dim` (total number of state variables, including latent ones
    like stochastic volatility).
    """

    @abstractmethod
    def simulate(
        self,
        S0: torch.Tensor,
        time_grid: torch.Tensor,
        n_paths: int,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Simulate paths on the given time grid.

        Parameters
        ----------
        S0 : Tensor, shape (d,) or (full_dim,)
            Initial state vector.
        time_grid : Tensor, shape (n_steps+1,)
            Ordered time points including t=0.  Example: [0, dt, 2dt, ..., T].
        n_paths : int
            Number of independent Monte Carlo paths.
        device : torch.device or None
            Device for computation.  If None, uses S0's device.

        Returns
        -------
        paths : Tensor, shape (n_paths, n_steps+1, full_dim)
            Simulated state vectors at each time point.
        """
        ...

    @property
    @abstractmethod
    def state_dim(self) -> int:
        """Dimension of the state visible to the payoff function.

        For plain GBM on d assets this equals d.
        For Heston on 1 asset this equals 1 (only S, not nu).
        """
        ...

    @property
    @abstractmethod
    def full_dim(self) -> int:
        """Total dimension of the state vector (including latent variables).

        For plain GBM this equals d.
        For Heston this equals 2 (S, nu).
        """
        ...

    def observable(self, paths: torch.Tensor) -> torch.Tensor:
        """Extract the payoff-relevant state from full paths.

        Parameters
        ----------
        paths : Tensor, shape (n_paths, n_steps+1, full_dim)

        Returns
        -------
        obs : Tensor, shape (n_paths, n_steps+1, state_dim)
            The first `state_dim` components (default behaviour).
            Override in subclasses if needed.
        """
        return paths[..., : self.state_dim]
