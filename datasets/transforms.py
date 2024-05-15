import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.transforms.functional import (
    _interpolation_modes_from_int,
    crop,
    center_crop,
)

from typing import List, Tuple
import math
from torchvision.transforms import functional as F

class CropBorder(nn.Module):
    def __init__(self, width, height) -> None:
        super().__init__()
        self.width = width
        self.height = height

    def forward(self, x: torch.Tensor):
        w, h = x.size
        return center_crop(x, [h - self.height, w - self.width])

def get_crop_params(img, scale: List[float], ratio: List[float]) -> Tuple[int, int, int, int]:
    """Get parameters for ``crop`` for a random sized crop.

    Args:
        img (PIL Image or Tensor): Input image.
        scale (list): range of scale of the origin size cropped
        ratio (list): range of aspect ratio of the origin aspect ratio cropped

    Returns:
        tuple: params (i, j, h, w) to be passed to ``crop`` for a random
        sized crop.
    """
    _, height, width = F.get_dimensions(img)
    area = height * width

    log_ratio = torch.log(torch.tensor(ratio))
    for _ in range(10):
        target_area = area * \
            torch.empty(1).uniform_(scale[0], scale[1]).item()
        aspect_ratio = torch.exp(torch.empty(1).uniform_(
            log_ratio[0], log_ratio[1])).item()

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        if 0 < w <= width and 0 < h <= height:
            i = torch.randint(0, height - h + 1, size=(1,)).item()
            j = torch.randint(0, width - w + 1, size=(1,)).item()
            return i, j, h, w

    # Fallback to central crop
    in_ratio = float(width) / float(height)
    if in_ratio < min(ratio):
        w = width
        h = int(round(w / min(ratio)))
    elif in_ratio > max(ratio):
        h = height
        w = int(round(h * max(ratio)))
    else:  # whole image
        w = width
        h = height
    i = (height - h) // 2
    j = (width - w) // 2
    return i, j, h, w


class CustomCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, **kwargs):
        for t in self.transforms:
            kwargs = t(**kwargs)
        return kwargs


class RandomResizedCrop:
    def __init__(
        self,
        size,
        scale,
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        interp_mode=2,
    ):
        self.interpolation = _interpolation_modes_from_int(interp_mode)
        self.label_interpolation = _interpolation_modes_from_int(0)
        self.size = size
        self.scale = scale
        self.ratio = ratio

    def __call__(self, **kwargs):
        res = {}
        for _, value in kwargs.items():
            img = value
            break
        i, j, h, w = get_crop_params(img, self.scale, self.ratio)
        for key, value in kwargs.items():
            if key == "label":
                res[key] = F.resized_crop(
                    img, i, j, h, w, self.size, self.label_interpolation)

            else:
                res[key] = F.resized_crop(
                    img, i, j, h, w, self.size, self.interpolation)

        return res


class RandomMirror:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, **kwargs):
        res = {}
        if torch.rand(1) < self.p:
            for key, value in kwargs.items():
                res[key] = F.hflip(value)
            return res
        else:
            return kwargs


if __name__ == '__main__':
    from PIL import Image
    img = Image.open("data/NYUDepthv2/seglabel/0.png")
    print(get_crop_params(img, scale=[0.2, 0.4], ratio=(0.2, 0.4)))
