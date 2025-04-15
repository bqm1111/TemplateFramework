from email.policy import default
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from datasets import get_dataset
import torch
from tqdm import tqdm
from collections import defaultdict

if __name__ == "__main__":
    config = OmegaConf.load("config/calculate_norm.yaml")

    train_dataset = get_dataset(config.dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=config.drop_last,
    )
# 
    psum = defaultdict(lambda: torch.tensor([0.0, 0.0, 0.0]))
    psum_sq = defaultdict(lambda: torch.tensor([0.0, 0.0, 0.0]))
    count = defaultdict(lambda: 0)
    for data in tqdm(train_loader):
        for key in data.keys():
            sample = data[key]
            if key == 'label':
                continue
            _, _, h, w = sample.shape
            count[key] += h * w
            psum[key] += sample.sum(axis=[0, 2, 3])
            psum_sq[key] += (sample**2).sum(axis=[0, 2, 3])

    # mean and std
    total_mean = dict()
    total_var = dict()
    total_std = dict()
    for key in psum.keys():
        total_mean[key] = psum[key] / count[key]
        total_var[key] = (psum_sq[key] / count[key]) - (total_mean[key] ** 2)
        total_std[key] = torch.sqrt(total_var[key])

    # output
    print("mean: " + str(total_mean))
    print("std:  " + str(total_std))
