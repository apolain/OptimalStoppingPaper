import random
from dataclasses import dataclass

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class TorchConfig:
    """Immutable (device, dtype, sim_dtype) propagated through the stack.

    Two dtypes serve different needs:

    - ``dtype`` (default float64): used for all *numerical* operations
      where precision matters — regression in LSMC, discounting,
      payoff comparisons, network weights & gradients.
    - ``sim_dtype`` (default float32): used for Monte Carlo path
      simulation, where float32 is sufficient and significantly faster
      (especially on GPU, where float64 can be 2–32× slower).

    Parameters
    ----------
    device : torch.device
    dtype : torch.dtype
        Precision dtype for numerics (default float64).
    sim_dtype : torch.dtype
        Simulation dtype for path generation (default float32).
    """

    device: torch.device
    dtype: torch.dtype
    sim_dtype: torch.dtype = torch.float32

    # convenience constructors -------------------------------------------

    @staticmethod
    def make(
        device: str | torch.device | None = None,
        dtype: str | torch.dtype = torch.float64,
        sim_dtype: str | torch.dtype = torch.float32,
    ) -> "TorchConfig":
        """Build a TorchConfig from user-friendly strings.

        Parameters
        ----------
        device : str, torch.device, or None
            ``None`` / ``"auto"`` → CUDA if available, else CPU.
        dtype : str or torch.dtype
            Numeric precision (default float64).
        sim_dtype : str or torch.dtype
            Simulation precision (default float32).
        """
        _map = {"float32": torch.float32, "float64": torch.float64}

        if device is None or device == "auto":
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            dev = torch.device(device)
        else:
            dev = device

        if isinstance(dtype, str):
            dtype = _map[dtype]
        if isinstance(sim_dtype, str):
            sim_dtype = _map[sim_dtype]

        return TorchConfig(device=dev, dtype=dtype, sim_dtype=sim_dtype)

    # tensor helpers (numeric dtype) --------------------------------------

    def tensor(self, data, **kwargs) -> torch.Tensor:
        """Shorthand: create a tensor on the right device and dtype."""
        return torch.tensor(data, device=self.device, dtype=self.dtype, **kwargs)

    def zeros(self, *shape, **kwargs) -> torch.Tensor:
        return torch.zeros(*shape, device=self.device, dtype=self.dtype, **kwargs)

    def ones(self, *shape, **kwargs) -> torch.Tensor:
        return torch.ones(*shape, device=self.device, dtype=self.dtype, **kwargs)

    def empty(self, *shape, **kwargs) -> torch.Tensor:
        return torch.empty(*shape, device=self.device, dtype=self.dtype, **kwargs)

    def randn(self, *shape, **kwargs) -> torch.Tensor:
        return torch.randn(*shape, device=self.device, dtype=self.dtype, **kwargs)

    def full(self, shape, fill_value, **kwargs) -> torch.Tensor:
        return torch.full(
            shape, fill_value, device=self.device, dtype=self.dtype, **kwargs
        )

    def linspace(self, start, end, steps, **kwargs) -> torch.Tensor:
        return torch.linspace(
            start, end, steps, device=self.device, dtype=self.dtype, **kwargs
        )

    # simulation helpers (sim_dtype) --------------------------------------

    def sim_randn(self, *shape, **kwargs) -> torch.Tensor:
        return torch.randn(*shape, device=self.device, dtype=self.sim_dtype, **kwargs)

    def sim_empty(self, *shape, **kwargs) -> torch.Tensor:
        return torch.empty(*shape, device=self.device, dtype=self.sim_dtype, **kwargs)


# Legacy helper kept for simple scripts that only need a device ----------


def get_device(device: str | None = None) -> torch.device:
    """Resolve device string to torch.device (legacy helper).

    Prefer ``TorchConfig.make(...)`` for new code.
    """
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
