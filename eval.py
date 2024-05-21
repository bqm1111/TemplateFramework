import os
import cv2
import argparse
import numpy as np

from omegaconf import OmegaConf
from engine.evaluator import Evaluator
from utils.logger import get_root_logger
from torch.utils.data import DataLoader
from engine import get_model

logger = get_root_logger()

parser = argparse.ArgumentParser()
parser.add_argument("--config")


if __name__ == '__main__':
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    model = get_model(config.model.name, eval=True, **config.model.params)
    segmentor = Evaluator(config, model)
    segmentor.run(
        "output_dir/semseg/nyuv2/checkpoint-500.pth")

