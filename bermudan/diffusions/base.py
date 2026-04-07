from abc import ABC, abstractmethod

import torch

from ..utils.seeds import TorchConfig


class Diffusion(ABC):
    """Base class for simulatable diffusion processes."""

    @abstractmethod
    def simulate(
        self,
        S0: torch.Tensor,
        time_grid: torch.Tensor,
        n_paths: int,
        cfg: TorchConfig | None = None,
    ) -> torch.Tensor:
        """Simulate paths on the given time grid.

        Parameters
        ----------
        S0 : Tensor, shape (d,) or (full_dim,)
            Initial state vector.
        time_grid : Tensor, shape (n_steps+1,)
            Ordered time points including t=0.
        n_paths : int
            Number of independent Monte Carlo paths.
        cfg : TorchConfig or None
            Device and dtype.  If None, inherits from S0.

        Returns
        -------
        paths : Tensor, shape (n_paths, n_steps+1, full_dim)
        """
        ...

    @property
    @abstractmethod
    def state_dim(self) -> int:
        """Dimension of the state visible to the payoff function."""
        ...

    @property
    @abstractmethod
    def full_dim(self) -> int:
        """Total dimension of the state vector (including latent variables)."""
        ...

    def observable(self, paths: torch.Tensor) -> torch.Tensor:
        """Extract the payoff-relevant state from full paths.

        Parameters
        ----------
        paths : Tensor, shape (n_paths, n_steps+1, full_dim)

        Returns
        -------
        obs : Tensor, shape (n_paths, n_steps+1, state_dim)
        """
        return paths[..., : self.state_dim]

    @staticmethod
    def _resolve_cfg(S0: torch.Tensor, cfg: TorchConfig | None) -> TorchConfig:
        """Resolve config: explicit cfg wins, otherwise infer from S0."""
        if cfg is not None:
            return cfg
        return TorchConfig(device=S0.device, dtype=S0.dtype)
