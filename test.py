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

logger = get_root_logger()

class BaseClass:
    def __init__(self):
        print("Do this first")
    
    def train(self):
        print("Train first")

class InheritClass(BaseClass):
    def __init__(self):
        super().__init__()
        print("Do this later")
    def train(self):
        print("Train second")


if __name__ == '__main__':
    a = InheritClass()
    a.train()
    