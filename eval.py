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
parser.add_argument("--show", action='store_true')
parser.add_argument("--epoch", type=int)
if __name__ == '__main__':
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    model = get_model(config.model.name, eval=True, **config.model.params)
    checkpoint_path = os.path.join("output_dir/", config.experiment_type, config.experiment_dataset,
                                   config.experiment_name, "checkpoint-" + str(args.epoch) + ".pth")
    segmentor = Evaluator(config, model, args.show)
    segmentor.run(checkpoint_path)
