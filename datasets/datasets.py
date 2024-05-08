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
    def __init__(self, root, split, transforms=None, depth_transform=None):
        self.root_dir = root
        self.split = split
        self.transforms = transforms
        self.depth_transform = depth_transform
        self.rgb_path = os.path.join(self.root_dir, "image")
        self.depth_path = os.path.join(self.root_dir, "depth")
        self.label_path = os.path.join(self.root_dir, "seglabel")
        self.depth_unfilled_path = os.path.join(self.root_dir, "Depth_unfilled")
        self.raw_depth_path = os.path.join(self.root_dir, "rawDepths")
        self.raw_depth_filled_path = os.path.join(self.root_dir, "rawDepths_filled")
        all_index_file = os.path.join(self.root_dir, split + "_data_idx.txt")
        if not os.path.exists(all_index_file):
            raise Exception(f"Split index file does not exist {all_index_file}")

        with open(all_index_file, "r") as f:
            self.all_index = [int(idx) for idx in f.readlines()]

    def __len__(self):
        return len(self.all_index)


class NYUv2Dataset(DepthDataset):
    def __init__(self, root, split, transforms=None, depth_transform=None):
        super(NYUv2Dataset, self).__init__(root, split, transforms, depth_transform)

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
    def __init__(self, root, split, transforms=None, depth_transform=None):
        super(SunRGBDDataset, self).__init__(root, split, transforms, depth_transform)

    def __getitem__(self, index):
        # Read all necessary types of image (rgb, depth, depth_anything, raw_depth)
        rgb = Image.open(
            os.path.join(self.rgb_path, str(index + 1).zfill(6) + ".jpg")
        ).convert("RGB")
        depth = Image.open(
            os.path.join(self.depth_path, str(index + 1).zfill(6) + ".png")
        )
        depth = np.array(depth)
        max_depth = np.max(depth)
        depth = (depth / max_depth * 255.0).astype(np.uint8)
        depth = Image.fromarray(np.stack((depth,) * 3, axis=-1))
        # raw_depth = np.load(os.path.join(
        #     self.raw_depth_path, str(index) + ".npy"))
        # Transform image
        if self.transforms is not None:
            rgb = self.transforms(rgb)
        if self.depth_transform is not None:
            depth = self.transforms(depth)
            # raw_depth = self.transforms(raw_depth)
        # Return output as a dictionary
        output = {"rgb": rgb, "depth": depth}
        if rgb is None and depth is None:
            logger.error("Receive NoneType")
            
        return output


class MNIST(torchvision.datasets.MNIST):
    pass
