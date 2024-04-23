from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from datasets import get_dataset
import torch
from tqdm import tqdm

if __name__ == '__main__':
    config = OmegaConf.load("config/calculate_norm.yaml")

    train_dataset = get_dataset(config.dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=config.drop_last,
    )
    psum = torch.tensor([0.0, 0.0, 0.0])
    psum_sq = torch.tensor([0.0, 0.0, 0.0])
    count = 0
    for sample in tqdm(train_loader):
        _, _, h, w = sample.shape
        count += h * w
        psum += sample.sum(axis=[0, 2, 3])
        psum_sq += (sample**2).sum(axis=[0, 2, 3])

    # mean and std
    total_mean = psum / count
    total_var = (psum_sq / count) - (total_mean**2)
    total_std = torch.sqrt(total_var)

    # output
    print("mean: " + str(total_mean))
    print("std:  " + str(total_std))
