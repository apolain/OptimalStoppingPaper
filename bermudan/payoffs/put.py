import torch

from .base import Payoff


class Put(Payoff):
    """Vanilla put payoff on a single underlying.

    g(s) = max(K - s, 0)

    Features: (log(s/K), g(s)/K, t/T, (T-t)/T).
    Using log-moneyness instead of s/K because GBM is log-normal.
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
                torch.log(s / self.K + 1e-8),  # log-moneyness
                payoff_val / self.K,  # normalised payoff
                t_val / T,  # normalised time
                (T - t_val) / T,  # time to maturity
            ],
            dim=-1,
        )

    @property
    def n_features(self) -> int:
        return 4
