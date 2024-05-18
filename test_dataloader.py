from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from datasets import get_dataset
import cv2
from utils.helper import show_pil_image

if __name__ == "__main__":
    config = OmegaConf.load("config/mae/dual_swin_small_normalized_target_depthanything.yaml")
    train_cfg = config.train
    dataset = get_dataset(train_cfg.dataset)
    train_loader = DataLoader(dataset, batch_size=1,
                              shuffle=False, num_workers=4)
    for sample in train_loader:
        numpy_image = sample["depth_anything"].squeeze().numpy()
        numpy_image = numpy_image.transpose(1, 2, 0)
        show_pil_image("img", numpy_image)
        if cv2.waitKey() == ord('q'):
            break

