import os
import cv2
import numpy as np
import time
from tqdm import tqdm
from timm.models.layers import to_2tuple

import torch
import multiprocessing as mp

from utils.logger import get_root_logger
from utils.misc import load_model_to_resume

logger = get_root_logger()


class Evaluator:
    def __init__(self, dataloader):
        self.dataloader = dataloader

    def run(self, model, model_file, log_file):
        with open(log_file, 'a') as f:
            model.load_state_dict(torch.load(model_file)["model"])
            result_line = self.single_process_evaluation()
            f.write('Model: ' + model + '\n')
            f.write(result_line)
            f.write('\n')
            f.flush()

    def single_process_evaluation(self):
        all_results = []
        for data in tqdm(self.dataloader):
            results_dict = self.func_per_iteration(data)
            all_results.append(results_dict)

        result_line = self.compute_metric(all_results)
        return result_line

    def compute_metric(self, results):
        raise NotImplementedError
    