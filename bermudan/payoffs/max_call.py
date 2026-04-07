import torch

from .base import Payoff


class MaxCall(Payoff):
    """Payoff on the maximum of d correlated assets.

    g(s^1, ..., s^d) = max(max_i s^i - K, 0)

    Features (d + 6 dimensions, matching the architecture that works):
      - log(s^i / K)  for each i          (d features: log-moneyness per asset)
      - log(max_i s^i / K)                (1: log-moneyness of the max)
      - log(mean_i s^i / K)               (1: average log-moneyness)
      - std(log(s^i / K))                 (1: dispersion of assets)
      - (top1 - top2) / K                 (1: gap between best two assets)
      - t / T                             (1: normalised time)
      - (T - t) / T                       (1: time to maturity)

    Why log-space: GBM dynamics are log-normal, so log-moneyness features
    are more natural and lead to better-conditioned gradients.

    Why permutation-invariant aggregates: for large d, the optimal policy
    depends primarily on summary statistics (max, mean, dispersion, gap)
    rather than the ordering of individual assets.
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

        logm = torch.log(S / self.K + eps)  # (M, d) log-moneyness
        max_S = S.max(dim=-1).values  # (M,)
        mean_S = S.mean(dim=-1)  # (M,)

        # Gap between top-2 assets
        if d >= 2:
            top2, _ = torch.topk(S, k=2, dim=-1)  # (M, 2)
            gap = (top2[:, 0] - top2[:, 1]) / self.K  # (M,)
        else:
            gap = torch.zeros(n, device=S.device, dtype=S.dtype)

        parts = [
            logm,  # (M, d)
            torch.log(max_S / self.K + eps).unsqueeze(-1),  # (M, 1)
            torch.log(mean_S / self.K + eps).unsqueeze(-1),  # (M, 1)
            logm.std(dim=-1, keepdim=True),  # (M, 1)
            gap.unsqueeze(-1),  # (M, 1)
            (t_val / T).unsqueeze(-1),  # (M, 1)
            ((T - t_val) / T).unsqueeze(-1),  # (M, 1)
        ]
        return torch.cat(parts, dim=-1)  # (M, d+6)

    @property
    def n_features(self) -> int:
        if self._d is None:
            raise RuntimeError(
                "n_features unknown until first call to features(). "
                "Pass d= to MaxCall constructor, or call features() first."
            )
        return self._d + 6
