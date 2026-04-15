import torch

from .base import Payoff


class MaxCall(Payoff):
    """Payoff on the maximum of d correlated assets.

    g(s^1, ..., s^d) = max(max_i s^i - K, 0)

    Features (d + 6 dimensions):
      - log(s^i / K)  for each i          (d features: log-moneyness per asset)
      - log(max_i s^i / K)                (1: log-moneyness of the max)
      - log(mean_i s^i / K)               (1: average log-moneyness)
      - std(log(s^i / K))                 (1: dispersion of assets)
      - (top1 - top2) / K                 (1: gap between best two assets)
      - t / T                             (1: normalised time)
      - (T - t) / T                       (1: time to maturity)
    """

    def __init__(self, K: float, d: int | None = None):
        super().__init__(K)
        self._d = d

    def __call__(self, S: torch.Tensor) -> torch.Tensor:
        max_S = S.max(dim=-1).values
        return torch.clamp(max_S - self.K, min=0.0)

    def features(
        self,
        S: torch.Tensor,
        t: torch.Tensor,
        T: float,
    ) -> torch.Tensor:
        n, d = S.shape
        if self._d is None:
            self._d = d
        t_val = t.expand(n) if t.dim() == 0 else t
        eps = 1e-8

        logm = torch.log(S / self.K + eps)
        max_S = S.max(dim=-1).values
        mean_S = S.mean(dim=-1)

        if d >= 2:
            top2, _ = torch.topk(S, k=2, dim=-1)
            gap = (top2[:, 0] - top2[:, 1]) / self.K
        else:
            gap = torch.zeros(n, device=S.device, dtype=S.dtype)

        parts = [
            logm,
            torch.log(max_S / self.K + eps).unsqueeze(-1),
            torch.log(mean_S / self.K + eps).unsqueeze(-1),
            logm.std(dim=-1, keepdim=True),
            gap.unsqueeze(-1),
            (t_val / T).unsqueeze(-1),
            ((T - t_val) / T).unsqueeze(-1),
        ]
        return torch.cat(parts, dim=-1)

    @property
    def n_features(self) -> int:
        if self._d is None:
            raise RuntimeError(
                "n_features unknown until first call to features(). "
                "Pass d= to MaxCall constructor, or call features() first."
            )
        return self._d + 6
