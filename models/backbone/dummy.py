import torch
import torch.nn as nn


class DummyModel(nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        