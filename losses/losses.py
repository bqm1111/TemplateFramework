import torch.nn as nn
import torch
import torch.nn.functional as F


class CustomLoss(nn.Module):
    def __init__(self):
        raise NotImplementedError

    def forward(self, x, y):
        raise NotImplementedError

# For KL divergence, see Appendix B in VAE paper or http://yunjey47.tistory.com/43


class KLDivergenceLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, log_var, mu):
        return (- 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()))

class BinaryCrossEntropy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x_reconst, x):
        return F.binary_cross_entropy(x_reconst, x, size_average=False)