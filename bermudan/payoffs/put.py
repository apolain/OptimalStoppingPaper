import torch

from .base import Payoff


class Put(Payoff):
    """Vanilla put payoff on a single underlying.

    g(s) = max(K - s, 0)

    Features for neural networks: (s/K, t/T, (T-t)/T).
    """

    def __call__(self, S: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        S : Tensor, shape (n_paths,) or (n_paths, 1)
        """
        s = S.squeeze(-1) if S.dim() == 2 else S
        return torch.clamp(self.K - s, min=0.0)

    def features(
        self,
        S: torch.Tensor,
        t: torch.Tensor,
        T: float,
    ) -> torch.Tensor:
        s = S.squeeze(-1) if S.dim() == 2 else S  # (M,)
        n = s.shape[0]
        t_val = t.expand(n) if t.dim() == 0 else t

        return torch.stack(
            [
                s / self.K,  # normalised spot
                t_val / T,  # normalised time
                (T - t_val) / T,  # time to maturity
            ],
            dim=-1,
        )  # (M, 3)

    @property
    def n_features(self) -> int:
        return 3
