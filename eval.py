import os
import cv2
import argparse
import numpy as np

from omegaconf import OmegaConf
import torch
import torch.nn as nn
from PIL import Image
from datasets import get_dataset
from engine import get_model
from engine.evaluator import Evaluator
from utils.logger import get_root_logger
from torch.utils.data import DataLoader

logger = get_root_logger()

parser = argparse.ArgumentParser()
parser.add_argument("--config")


class SegEvaluator(Evaluator):
    def func_per_iteration(self, data, device):
        pass


if __name__ == '__main__':
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    val_cfg = config.val
    model = get_model(config.model.name, **config.model.params)

    val_dataset = get_dataset(val_cfg.dataset)
    val_loader = DataLoader(val_dataset, batch_size=val_cfg.batch_size, shuffle=False,
                            num_workers=val_cfg.num_workers, drop_last=val_cfg.drop_last)

    with torch.no_grad():
        segmentor = SegEvaluator()
        segmentor.run()


