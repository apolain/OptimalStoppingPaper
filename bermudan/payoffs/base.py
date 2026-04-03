from abc import ABC, abstractmethod

import torch


class Payoff(ABC):
    """Base class for option payoff functions.

    A payoff maps the observable state S ∈ R^d to a non-negative scalar.
    Subclasses must implement `__call__` and `features`.
    """

    def __init__(self, K: float):
        self.K = K

    @abstractmethod
    def __call__(self, S: torch.Tensor) -> torch.Tensor:
        """Evaluate the payoff.

        Parameters
        ----------
        S : Tensor, shape (n_paths, d) or (n_paths,)
            Observable asset prices at exercise time.

        Returns
        -------
        payoff : Tensor, shape (n_paths,)
            Non-negative payoff values.
        """
        ...

    @abstractmethod
    def features(
        self,
        S: torch.Tensor,
        t: torch.Tensor,
        T: float,
    ) -> torch.Tensor:
        """Build enriched feature vector for neural network input.

        This produces the input that neural networks (DOS / PG) will see.
        The feature set depends on the payoff structure: e.g. for max-call,
        it includes max_i S^i and moneyness alongside time features.

        Parameters
        ----------
        S : Tensor, shape (n_paths, d)
            Observable asset prices.
        t : Tensor, shape (n_paths,) or scalar
            Current time.
        T : float
            Maturity.

        Returns
        -------
        phi : Tensor, shape (n_paths, n_features)
        """
        ...

    @property
    @abstractmethod
    def n_features(self) -> int:
        """Number of features returned by `features`."""
        ...
