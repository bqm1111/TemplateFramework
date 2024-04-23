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

if __name__ == "__main__":
    config = OmegaConf.load("config/mae.yaml")
    train_cfg = config.train
    dataset = get_dataset(train_cfg.dataset)
    train_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    transform = T.ToPILImage()
    for sample in train_loader:
        numpy_image = sample.squeeze().numpy()
        pil_image = F.to_pil_image(numpy_image.transpose(1, 2, 0).astype(np.uint8))
        # img = transform(numpy_image.transpose(1, 2, 0))
        pil_image.show()
        # numpy_image = sample.squeeze().numpy()
        # print(numpy_image.shape)
        # pil_image = Image.fromarray(numpy_image)
        # pil_image.show()
        # cv2.imshow("img", np.array(numpy_image).transpose(1, 2, 0))
        # cv2.waitKey()
