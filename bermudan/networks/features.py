import torch

from ..payoffs.base import Payoff


def build_features(
    payoff: Payoff,
    S: torch.Tensor,
    t: torch.Tensor | float,
    T: float,
) -> torch.Tensor:
    """Build feature vector for neural network input.

    Parameters
    ----------
    payoff : Payoff
        Payoff object whose `features` method defines the mapping.
    S : Tensor, shape (n_paths, state_dim)
        Observable state at current time.
    t : Tensor or float
        Current time (scalar or per-path).
    T : float
        Maturity.

    Returns
    -------
    phi : Tensor, shape (n_paths, n_features)
    """
    if isinstance(t, (int, float)):
        t = torch.tensor(t, device=S.device, dtype=S.dtype)
    return payoff.features(S, t, T)
