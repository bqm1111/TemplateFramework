import time
import numpy as np
import torch
import torch.nn.functional as F
import os.path as osp
import os


class Timer():
    def __init__(self) -> None:
        self.start_time = 0.0
        self.end_time = 0.0
        self.start()

    def start(self):
        self.start_time = time.time()

    def end(self, ms=False, clear=False):
        self.end_time = time.time()
        if ms:
            duration = int((self.end_time - self.start_time) * 1000)
        else:
            duration = int(self.end_time - self.start_time)

        if clear:
            self.start_time = time.time()

        return duration


class Average_Meter:
    def __init__(self, keys):
        self.keys = keys
        self.clear()

    def add(self, dic):
        for key, value in dic.items():
            self.data_dic[key].append(value)

    def get(self, keys=None, clear=False):
        if keys is None:
            keys = self.keys

        dataset = {}
        for key in keys:
            dataset[key] = float(np.mean(self.data_dic[key]))

        if clear:
            self.clear()

        return dataset

    def clear(self):
        self.data_dic = {key: [] for key in self.keys}
