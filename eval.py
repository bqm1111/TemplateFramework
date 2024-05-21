import os
import cv2
import argparse
import numpy as np

from omegaconf import OmegaConf
from engine.evaluator import Evaluator
from utils.logger import get_root_logger
from torch.utils.data import DataLoader

logger = get_root_logger()

parser = argparse.ArgumentParser()
parser.add_argument("--config")


if __name__ == '__main__':
    args = parser.parse_args()
    config = OmegaConf.load(args.config)

    segmentor = Evaluator(config)
    segmentor.run("output_dir/semseg/dual_swin_small_normalized_target_origin/checkpoint-500.pth")


