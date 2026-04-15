import torch

from .base import Payoff


class Put(Payoff):
    """Vanilla put payoff on a single underlying.

    g(s) = max(K - s, 0)

    Features: (log(s/K), g(s)/K, t/T, (T-t)/T).
    """

    def __call__(self, S: torch.Tensor) -> torch.Tensor:
        s = S.squeeze(-1) if S.dim() == 2 else S
        return torch.clamp(self.K - s, min=0.0)

    def features(
        self,
        S: torch.Tensor,
        t: torch.Tensor,
        T: float,
    ) -> torch.Tensor:
        s = S.squeeze(-1) if S.dim() == 2 else S
        n = s.shape[0]
        t_val = t.expand(n) if t.dim() == 0 else t

        payoff_val = torch.clamp(self.K - s, min=0.0)

        return torch.stack(
            [
                torch.log(s / self.K + 1e-8),
                payoff_val / self.K,
                t_val / T,
                (T - t_val) / T,
            ],
            dim=-1,
        )

    @property
    def n_features(self) -> int:
        return 4
