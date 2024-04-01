import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import torchvision


class DepthDataset(Dataset):
    def __init__(self, root, split, transforms=None, target_transform=None):
        self.root_dir = root
        self.split = split
        self.transforms = transforms
        self.rgb_path = os.path.join(self.root_dir, "RGB")
        self.depth_path = os.path.join(self.root_dir, "Depth")
        self.label_path = os.path.join(self.root_dir, "Label")
        self.depth_unfilled_path = os.path.join(
            self.root_dir, "Depth_unfilled")
        self.raw_depth_path = os.path.join(self.root_dir, "rawDepths")
        self.raw_depth_filled_path = os.path.join(
            self.root_dir, "rawDepths_filled")
        all_index_file = os.path.join(self.root_dir, split + "_mod.txt")

        if not os.path.exists(all_index_file):
            raise Exception(
                f"Split index file does not exist {all_index_file}")

        with open(all_index_file, "r") as f:
            self.all_index = [int(idx) for idx in f.readlines()]

    def __getitem__(self, index):
        rgb = self._open_image(os.path.join(
            self.rgb_path, str(index) + ".jpg"))
        if self.transforms is not None:
            for transform in self.transforms:
                rgb = transform(rgb)

        return rgb

    @staticmethod
    def _open_image(filepath, mode=cv2.IMREAD_COLOR, dtype=None):
        img = np.array(cv2.imread(filepath, mode), dtype=dtype)
        return img

    def __len__(self):
        return len(self.all_index)


class MNIST(torchvision.datasets.MNIST):
    pass
