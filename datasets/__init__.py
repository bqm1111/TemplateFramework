from .transforms import CustomRandomResizedCrop, CropBorder
from omegaconf.dictconfig import DictConfig
from .datasets import DepthDataset, MNIST, NYUv2Dataset, SunRGBDDataset
from utils.logger import get_root_logger
import torchvision.transforms as T

ALL_TRANSFORM = {
    "crop_border": CropBorder,
    "resize": T.Resize,
    "to_tensor": T.ToTensor,
    "RandomResizedCrop": CustomRandomResizedCrop,
    "RandomHorizontalFlip": T.RandomHorizontalFlip,
    "normalize": T.Normalize,
}

ALL_DATASETS = {
    "nyuv2": NYUv2Dataset,
    "sunrgbd": SunRGBDDataset,
    "mnist": MNIST,
}
logger = get_root_logger()


def get_dataset(cfg):
    if cfg is None:
        return None
    name = cfg.name
    if name not in ALL_DATASETS:
        logger.warning(
            "{name} is not supported, please implement it first.".format(name=name)
        )
        return None

    transform = get_transform(cfg.transforms)
    depth_transform = get_transform(cfg.depth_transforms)

    return ALL_DATASETS[name](
        **cfg.params, transforms=transform, depth_transform=depth_transform
    )


def get_transform(transforms: DictConfig):
    transform_list = []
    if transforms is None:
        return None
    for name in transforms.keys():
        assert name in ALL_TRANSFORM, (
            "{T_name} is not supported transform, please implement it and add it to "
            "ALL_TRANSFORM first.".format(T_name=name)
        )
        if transforms[name].params is not None:
            transform_list.append(ALL_TRANSFORM[name](**transforms[name].params))
        else:
            transform_list.append(ALL_TRANSFORM[name]())
    return T.Compose(transform_list)


class Iterator:
    def __init__(self, loader):
        self.loader = loader
        self.init()

    def init(self):
        self.iterator = iter(self.loader)

    def get(self):
        try:
            data = next(self.iterator)
        except StopIteration:
            self.init()
            data = next(self.iterator)
        return data
