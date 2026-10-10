"""The original fully connected neural vector field."""

import torch
from torch import nn
from torch.nn import functional as F


class NeuralVectorField(nn.Module):
    """A vector field with `depth` hidden layers and a linear output layer."""

    def __init__(
        self,
        input_dim: int,
        depth: int,
        latent_dim: int,
        activation: bool = True,
        *,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        layers = [nn.Linear(input_dim, latent_dim, dtype=dtype)]
        layers.extend(nn.Linear(latent_dim, latent_dim, dtype=dtype) for _ in range(depth - 1))
        layers.append(nn.Linear(latent_dim, input_dim, dtype=dtype))
        self.net = nn.ModuleList(layers)
        self.relu = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.net):
            x = layer(x)
            if i < len(self.net) - 1:
                x = F.relu(x) if self.relu else F.gelu(x)
        return x
