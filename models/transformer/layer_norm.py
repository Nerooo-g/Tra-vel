import torch
from torch import nn


class LayerNorm(nn.Module):
    """
    Layer normalization module.

    Normalizes the features of a batch of inputs across a certain axis.

    Args:
        d_model (int): Size of input features.
        eps (float): Small value to avoid division by zero.
        bias (bool): Whether to include a trainable bias parameter.

    Attributes:
        gamma (nn.Parameter): trainable scale parameter
        beta (nn.Parameter): trainable location parameter

    """
    def __init__(self, d_model, eps=1e-4, bias=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = nn.Parameter(torch.ones(d_model))
        if bias:
            self.beta = nn.Parameter(torch.zeros(d_model))
        else:
            self.register_parameter('beta', None)
        self.eps = eps

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (N, T, d_model)

        Returns:
            Tensor: Layer normalized input
        """
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)

        out = (x - mean) / torch.sqrt(var + self.eps)
        if self.beta is not None:
            out = self.gamma * out + self.beta
        else:
            out = self.gamma * out
        return out
