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
        label_transforms=None,
        common_transforms=None,
    ):
        self.root_dir = root
        self.split = split
        self.transforms = transforms
        self.target_tranforms = target_transforms
        self.depth_transforms = depth_transforms
        self.label_transforms = label_transforms
        self.common_transforms = common_transforms
        self.rgb_path = os.path.join(self.root_dir, "image")
        self.depth_path = os.path.join(self.root_dir, "depth")
        self.label_path = os.path.join(self.root_dir, "seglabel")
        self.raw_depth_anything_path = os.path.join(
            self.root_dir, "rawDepthAnything")
        all_index_file = os.path.join(self.root_dir, split + ".txt")
        if not os.path.exists(all_index_file):
            raise Exception(f"File does not exist {all_index_file}")

        with open(all_index_file, "r") as f:
            self.all_index = [int(idx) for idx in f.readlines()]

    @staticmethod
    def convert_raw_depth_to_3_channels_img(depth):
        max_depth = np.max(depth)
        depth = (depth / max_depth * 255.0).astype(np.uint8)
        depth = Image.fromarray(np.stack((depth,) * 3, axis=-1))
        return depth

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
        label_transforms=None,
        common_transforms=None,
    ):
        super(NYUv2Dataset, self).__init__(
            root,
            split,
            transforms,
            target_transforms,
            depth_transforms,
            label_transforms,
            common_transforms,
        )

    def __getitem__(self, index):
        # Read all necessary types of image (rgb, depth, depth_anything, raw_depth)
        possible_output = {"rgb": self.transforms, "depth": self.depth_transforms,
                           "depth_anything": self.target_tranforms, "label": self.label_transforms}
        rgb = Image.open(
            os.path.join(self.rgb_path, str(index) + ".jpg")
        ).convert("RGB")
        depth = np.load(os.path.join(self.depth_path, str(index) + ".npy"))
        depth = self.convert_raw_depth_to_3_channels_img(depth)
        if os.path.exists(self.raw_depth_anything_path):
            raw_depth_anything = np.load(os.path.join(
                self.raw_depth_anything_path, str(index) + ".npy"))
            raw_depth_anything = self.convert_raw_depth_to_3_channels_img(
                raw_depth_anything)
        else:
            raw_depth_anything = depth
        label = cv2.imread(os.path.join(
            self.label_path, str(index) + ".png"), cv2.IMREAD_GRAYSCALE)
        label = label - 1
        all_output = {"rgb": rgb, "depth": depth,
                  "depth_anything": raw_depth_anything, "label": label}
        output = {}
        for key, value in possible_output.items():
            if value is not None:
                output[key] = all_output[key] 
        if self.common_transforms is not None:
            output = self.common_transforms(**output)

        # Transform image
        if self.transforms is not None:
            output["rgb"] = self.transforms(output["rgb"])

        if self.depth_transforms is not None:
            output["depth"] = self.depth_transforms(output["depth"])

        if self.target_tranforms is not None:
            output["depth_anything"] = self.target_tranforms(
                output["depth_anything"])

        # Return output as a dictionary
        if rgb is None and depth is None:
            logger.error("Receive NoneType")

        return output


class SunRGBDDataset(DepthDataset):
    def __init__(
        self,
        root,
        split,
        transforms=None,
        target_transforms=None,
        depth_transforms=None,
        label_transforms=None,
        common_transforms=None,
    ):
        super(SunRGBDDataset, self).__init__(
            root,
            split,
            transforms,
            target_transforms,
            depth_transforms,
            label_transforms,
            common_transforms,
        )

    def __getitem__(self, index):
        # Read all necessary types of image (rgb, depth, depth_anything, raw_depth)
        rgb = Image.open(
            os.path.join(self.rgb_path, str(index + 1).zfill(6) + ".jpg")
        ).convert("RGB")
        depth = np.array(
            Image.open(os.path.join(self.depth_path,
                       str(index + 1).zfill(6) + ".png"))
        )

        raw_depth_anything = np.load(
            os.path.join(self.raw_depth_anything_path,
                         str(index + 1).zfill(6) + ".npy")
        )
        depth = self.convert_raw_depth_to_3_channels_img(depth)
        raw_depth_anything = self.convert_raw_depth_to_3_channels_img(
            raw_depth_anything
        )
        label = cv2.imread(os.path.join(
            self.label_path, str(index + 1).zfill(6) + ".png"), cv2.IMREAD_GRAYSCALE)

        output = {"rgb": rgb, "depth": depth,
                  "depth_anything": raw_depth_anything}
        if self.common_transforms is not None:
            output = self.common_transforms(**output)

        # Transform image
        if self.transforms is not None:
            output["rgb"] = self.transforms(output["rgb"])

        if self.depth_transforms is not None:
            output["depth"] = self.depth_transforms(output["depth"])

        if self.target_tranforms is not None:
            output["depth_anything"] = self.target_tranforms(
                output["depth_anything"])

        if self.label_transforms is not None:
            output["label"] = self.label_transforms(output["label"])

        # Return output as a dictionary
        if rgb is None and depth is None:
            logger.error("Receive NoneType")
        return output


if __name__ == "__main__":
    raw = np.load("data/sunrgbd_trainval/rawDepthAnything/000128.npy")
    print(raw)
