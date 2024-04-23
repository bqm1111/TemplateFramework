import argparse
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from datasets import get_dataset
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
import torch

config = OmegaConf.load("config/calculate_norm.yaml")

train_dataset = get_dataset(config.dataset)
train_loader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=config.num_workers,
    drop_last=config.drop_last,
)
train_loader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=config.num_workers,
    drop_last=config.drop_last,
)
psum = torch.tensor([0.0, 0.0, 0.0])
psum_sq = torch.tensor([0.0, 0.0, 0.0])

for sample in train_loader:
    psum += sample.sum(axis=[0, 2, 3])
    psum_sq += (sample**2).sum(axis=[0, 2, 3])

count = len(train_loader) * 640 * 480 * config.batch_size
# mean and std
total_mean = psum / count
total_var = (psum_sq / count) - (total_mean**2)
total_std = torch.sqrt(total_var)

# output
print("mean: " + str(total_mean))
print("std:  " + str(total_std))
