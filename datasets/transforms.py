import torchvision.transforms as T
import torch.nn as nn
from torchvision.transforms.functional import InterpolationMode, _interpolation_modes_from_int


class CustomRandomResizedCrop(T.RandomResizedCrop):
    def __init__(self, size, scale=..., ratio=(3.0 / 4.0, 4.0 / 3.0), interpolation=InterpolationMode.BILINEAR, interp_mode=2):
        super().__init__(size, scale, ratio, interpolation)
        self.interpolation = _interpolation_modes_from_int(interp_mode)


class CustomTransform(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        pass
