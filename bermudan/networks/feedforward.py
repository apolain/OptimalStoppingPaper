import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """Fully-connected network with configurable depth and width.

    Parameters
    ----------
    input_dim : int
        Size of the input feature vector.
    hidden_dims : list[int]
        Widths of each hidden layer.  E.g. [64, 64] for two layers of 64.
    output_dim : int, default 1
        Size of the output.
    activation : str, default "relu"
        Activation for hidden layers: "relu" or "tanh".
    output_activation : str or None, default None
        Activation on the final layer.  "sigmoid" for probability output.
    batch_norm_input : bool, default False
        If True, apply BatchNorm1d on the raw input before the first
        linear layer.  Normalises heterogeneous feature scales (prices
        ~100, times ~0.3) and is critical for DOS convergence.
        Safely skips normalisation for single-sample batches.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int = 1,
        activation: str = "relu",
        output_activation: str | None = None,
        batch_norm_input: bool = False,
    ):
        super().__init__()

        self._bn_input = nn.BatchNorm1d(input_dim) if batch_norm_input else None

        act_fn = {"relu": nn.ReLU, "tanh": nn.Tanh}[activation]

        layers: list[nn.Module] = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(act_fn())
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))

        if output_activation == "sigmoid":
            layers.append(nn.Sigmoid())
        elif output_activation is not None:
            raise ValueError(f"Unknown output activation: {output_activation}")

        self.net = nn.Sequential(*layers)
        self._n_params = sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._bn_input is not None and x.shape[0] > 1:
            x = self._bn_input(x)
        return self.net(x)

    @property
    def n_params(self) -> int:
        return self._n_params
