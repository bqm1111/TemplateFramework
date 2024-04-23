import torch
from datasets.datasets import DepthDataset
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from datasets import get_dataset
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as F
from utils.helper import show_pil_image
if __name__ == "__main__":
    config = OmegaConf.load("config/mae.yaml")
    train_cfg = config.train
    dataset = get_dataset(train_cfg.dataset)
    train_loader = DataLoader(dataset, batch_size=1,
                              shuffle=False, num_workers=4)
    for sample in train_loader:
        numpy_image = sample.squeeze().numpy()
        numpy_image = numpy_image.transpose(1, 2, 0)
        # print(numpy_image)
        show_pil_image("img", numpy_image)
        if cv2.waitKey() == ord('q'):
            break
