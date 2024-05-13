import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import torchvision
from PIL import Image, ImageOps
from utils.logger import get_root_logger
from utils.helper import show_pil_image

logger = get_root_logger()


class DepthDataset(Dataset):
    def __init__(
        self,
        root,
        split,
        transforms=None,
        target_transforms=None,
        depth_transforms=None,
        common_transforms=None,
    ):
        self.root_dir = root
        self.split = split
        self.transforms = transforms
        self.target_tranforms = target_transforms
        self.depth_transforms = depth_transforms
        self.common_transforms = common_transforms
        self.rgb_path = os.path.join(self.root_dir, "image")
        self.depth_path = os.path.join(self.root_dir, "depth")
        self.label_path = os.path.join(self.root_dir, "seglabel")
        self.raw_depth_anything_path = os.path.join(self.root_dir, "rawDepthAnything")
        all_index_file = os.path.join(self.root_dir, split + "_data_idx.txt")
        if not os.path.exists(all_index_file):
            raise Exception(f"Split index file does not exist {all_index_file}")

        with open(all_index_file, "r") as f:
            self.all_index = [int(idx) for idx in f.readlines()]

    def __len__(self):
        return len(self.all_index)


class NYUv2Dataset(DepthDataset):
    def __init__(
        self,
        root,
        split,
        transforms=None,
        target_transforms=None,
        depth_transforms=None,
        common_transforms=None,
    ):
        super(NYUv2Dataset, self).__init__(
            root,
            split,
            transforms,
            target_transforms,
            depth_transforms,
            common_transforms,
        )

    def __getitem__(self, index):
        # Read all necessary types of image (rgb, depth, depth_anything, raw_depth)
        rgb = Image.open(os.path.join(self.rgb_path, str(index) + ".jpg")).convert(
            "RGB"
        )
        raw_depth = np.load(os.path.join(self.raw_depth_path, str(index) + ".npy"))
        # Transform image
        if self.transforms is not None:
            rgb = self.transforms(rgb)
            # raw_depth = self.transforms(raw_depth)
        # Return output as a dictionary
        output = {"rgb": None, "depth": None, "depth_anything": None}
        if rgb is None:
            logger.error("Receive NoneType")
        return rgb


class SunRGBDDataset(DepthDataset):
    def __init__(
        self,
        root,
        split,
        transforms=None,
        target_transforms=None,
        depth_transforms=None,
        common_transforms=None,
    ):
        super(SunRGBDDataset, self).__init__(
            root,
            split,
            transforms,
            target_transforms,
            depth_transforms,
            common_transforms,
        )

    def __getitem__(self, index):
        # Read all necessary types of image (rgb, depth, depth_anything, raw_depth)
        rgb = Image.open(
            os.path.join(self.rgb_path, str(index + 1).zfill(6) + ".jpg")
        ).convert("RGB")
        depth = np.array(
            Image.open(os.path.join(self.depth_path, str(index + 1).zfill(6) + ".png"))
        )
        raw_depth_anything = np.load(
            os.path.join(self.raw_depth_anything_path, str(index + 1).zfill(6) + ".npy")
        )
        depth = self.convert_raw_depth_to_3_channels_img(depth)
        raw_depth_anything = self.convert_raw_depth_to_3_channels_img(
            raw_depth_anything
        )
        if self.common_transforms is not None:
            rgb = self.common_transforms(rgb)
            depth = self.common_transforms(depth)
            raw_depth_anything = self.common_transforms(raw_depth_anything)
        # Transform image
        if self.transforms is not None:
            rgb = self.transforms(rgb)

        if self.depth_transforms is not None:
            depth = self.depth_transforms(depth)

        if self.target_tranforms is not None:
            raw_depth_anything = self.target_tranforms(raw_depth_anything)

        # Return output as a dictionary
        output = {"rgb": rgb, "depth": depth, "depth_anything": raw_depth_anything}
        if rgb is None and depth is None:
            logger.error("Receive NoneType")
        if self.depth_transforms is not None:
            return output
        else:
            return rgb

    @staticmethod
    def convert_raw_depth_to_3_channels_img(depth):
        max_depth = np.max(depth)
        depth = (depth / max_depth * 255.0).astype(np.uint8)
        depth = Image.fromarray(np.stack((depth,) * 3, axis=-1))
        return depth


class MNIST(torchvision.datasets.MNIST):
    pass


if __name__ == "__main__":
    raw = np.load("data/sunrgbd_trainval/rawDepthAnything/000128.npy")
    print(raw)
