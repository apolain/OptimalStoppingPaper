from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch

from ..options.bermudan import BermudanOption


@dataclass
class PricingResult:
    """Unified output of any pricing method.

    Attributes
    ----------
    price : float
        Estimated option price V(0, S0).
    std : float
        Standard deviation of the Monte Carlo estimate (0 if not applicable).
    elapsed : float
        Wall-clock time in seconds.
    info : dict
        Method-specific extras (e.g. trained networks, loss history,
        exercise boundaries, continuation values).
    """

    price: float = 0.0
    std: float = 0.0
    elapsed: float = 0.0
    info: dict[str, Any] = field(default_factory=dict)


class PricingMethod(ABC):
    """Base class for all pricing methods (LSMC, DOS, PG)."""

    @abstractmethod
    def price(
        self,
        option: BermudanOption,
        S0: torch.Tensor,
        n_paths: int,
        device: torch.device | None = None,
        **kwargs,
    ) -> PricingResult:
        """Compute the Bermudan option price.

        Parameters
        ----------
        option : BermudanOption
            Full problem specification.
        S0 : Tensor
            Initial state vector.
        n_paths : int
            Number of Monte Carlo paths.
        device : torch.device or None
            Computation device.  None → auto-detect.

        Returns
        -------
        PricingResult
        """
        ...
