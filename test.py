from torchvision.transforms import ToTensor
from torchvision import datasets
from torch import nn
import torch
from omegaconf import OmegaConf

from datasets.datasets import DepthDataset
from torch.utils.data import DataLoader
import os
import cv2
from losses import get_losses
import numpy as np
from copy import deepcopy
from utils.logger import get_root_logger
import torch.nn as nn
from timm.optim import optim_factory
from models.segmentors import EncoderDecoder
from engine import get_model
logger = get_root_logger()


if __name__ == '__main__':
    # img = cv2.imread("/home/sherlock/Pictures/segmentation/sunrgbd/sunrgbd_trainval/seglabel/000009.png", cv2.IMREAD_GRAYSCALE)
    # print(np.min(img))
    # cv2.imshow("img", img)
    # cv2.waitKey()
    config = OmegaConf.load(
        "config/semseg/nyuv2/dual_swin_small_normalized_target_origin_nyuv2_frozen_0.yaml")
    net = get_model(config.model.name, eval=True, **config.model.params)
    h = 480
    w = 640
    y = net(torch.ones(1, 3, h, w).float(), torch.ones(
        1, 3, h, w).float(), torch.randint(0, 40, (1, h, w)).long())
    print(y.shape)
