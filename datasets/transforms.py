import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.transforms.functional import (
    InterpolationMode,
    _interpolation_modes_from_int,
    crop,
    center_crop,
)


class CustomRandomResizedCrop(T.RandomResizedCrop):
    def __init__(
        self,
        size,
        scale=...,
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        interpolation=InterpolationMode.BILINEAR,
        interp_mode=2,
    ):
        super().__init__(size, scale, ratio, interpolation)
        self.interpolation = _interpolation_modes_from_int(interp_mode)


class CropBorder(nn.Module):
    def __init__(self, width, height) -> None:
        super().__init__()
        self.width = width
        self.height = height

    def forward(self, x: torch.Tensor):
        w, h = x.size
        return center_crop(x, [h - self.height, w - self.width])


class CustomTransform(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        pass


if __name__ == "__main__":
    import cv2
    import numpy as np
    from PIL import Image
    filename = "data/NYUDepthv2/RGB/7.jpg"
    img = Image.open(filename).convert("RGB")
    print(img.size)
    crop_func = CropBorder(32, 32)
    out = crop_func(img)
    out.show()
    # Convert the numpy array to a cv2 image
    # cv2_image = np.transpose(numpy_image, (1, 2, 0))
    # # cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    # cv2.imshow("out", cv2_image)
    # cv2.waitKey()
