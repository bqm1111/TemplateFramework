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
    config = OmegaConf.load(
        "config/semseg/dual_swin_small_normalized_target_origin.yaml")
    net = get_model(config.model.name, **config.model.params)
    y = net(torch.ones(1, 3, 224, 224).float(), torch.ones(
        1, 3, 224, 224).float(), torch.randint(0, 40, (1, 224, 224)).long())
