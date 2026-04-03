import torch

from .base import Payoff


class MaxCall(Payoff):
    """Payoff on the maximum of d correlated assets.

    g(s^1, ..., s^d) = max(max_i s^i - K, 0)

    Features for neural networks
    -----------------------------
    Compact representation that scales to high d:
        (max_i s^i / K,  (max_i s^i - K) / K,  t/T,  (T-t)/T)

    This avoids feeding all d raw prices to the network in the 100-d case.
    The individual prices are not needed because the payoff and optimal
    policy depend primarily on the running maximum and time.
    """

    def __call__(self, S: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        S : Tensor, shape (n_paths, d)
        """
        max_S = S.max(dim=-1).values  # (M,)
        return torch.clamp(max_S - self.K, min=0.0)

    def features(
        self,
        S: torch.Tensor,
        t: torch.Tensor,
        T: float,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        S : Tensor, shape (n_paths, d)
        """
        n = S.shape[0]
        t_val = t.expand(n) if t.dim() == 0 else t

        max_S = S.max(dim=-1).values  # (M,)

        return torch.stack(
            [
                max_S / self.K,  # normalised max
                (max_S - self.K) / self.K,  # moneyness
                t_val / T,  # normalised time
                (T - t_val) / T,  # time to maturity
            ],
            dim=-1,
        )  # (M, 4)

    @property
    def n_features(self) -> int:
        return 4
