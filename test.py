from omegaconf import OmegaConf

from datasets.datasets import DepthDataset
from torch.utils.data import DataLoader
import os
import cv2
from losses import get_losses
config = OmegaConf.load("config/config.yaml")

# dataset = DepthDataset(**config.train.dataset.params)

# dataloader = DataLoader(dataset=dataset, batch_size=2)

# for img in dataloader:
#     print(img.shape)
#     cv2.imshow("img", img)
#     if cv2.waitKey() == ord('q'):
#         break

train_cfg = config.train
losses = get_losses(train_cfg.losses)
for item in losses.items():
    print(item[0])