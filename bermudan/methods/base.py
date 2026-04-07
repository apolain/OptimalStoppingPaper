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
        Standard deviation of the Monte Carlo estimate.
    elapsed : float
        Wall-clock time in seconds.
    info : dict
        Method-specific extras (trained networks, loss history, etc.).
    """

    price: float = 0.0
    std: float = 0.0
    elapsed: float = 0.0
    info: dict[str, Any] = field(default_factory=dict)


class PricingMethod(ABC):
    """Base class for all pricing methods (LSMC, DOS, PG).

    The computation device and dtype are read from ``option.cfg``.
    """

    @abstractmethod
    def price(
        self,
        option: BermudanOption,
        S0: torch.Tensor,
        n_paths: int,
        **kwargs,
    ) -> PricingResult:
        """Compute the Bermudan option price.

        Parameters
        ----------
        option : BermudanOption
            Full problem specification (includes cfg with device/dtype).
        S0 : Tensor
            Initial state vector.
        n_paths : int
            Number of Monte Carlo paths.

        Returns
        -------
        PricingResult
        """
        ...
