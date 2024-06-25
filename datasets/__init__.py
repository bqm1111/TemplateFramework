from .transforms import CropBorder, CustomCompose, RandomMirror, RandomResizedCrop
from .semseg.preprocess import TrainPre
from omegaconf.dictconfig import DictConfig
from .datasets import NYUv2Dataset, SunRGBDDataset
from utils.logger import get_root_logger
import torchvision.transforms as T
from .class_names import nyuv2_classname, sunrgbd_classname
ALL_CLASS_NAMES = {
    'nyuv2': nyuv2_classname,
    'sunrgbd': sunrgbd_classname
}


def get_classname(dataset_name):
    return ALL_CLASS_NAMES[dataset_name]()


ALL_TRANSFORM = {
    "crop_border": CropBorder,
    "resize": T.Resize,
    "to_tensor": T.ToTensor,
    "RandomHorizontalFlip": T.RandomHorizontalFlip,
    "normalize": T.Normalize,
    "to_pil": T.ToPILImage
}
ALL_CUSTOM_TRANSFORM = {
    "random_mirror": RandomMirror,
    "random_resized_crop": RandomResizedCrop,
    "semseg_transform": TrainPre
}

ALL_DATASETS = {
    "nyuv2": NYUv2Dataset,
    "sunrgbd": SunRGBDDataset,
}
logger = get_root_logger()


def get_dataset(cfg):
    name = cfg.name
    if name not in ALL_DATASETS:
        logger.warning(
            "{name} is not supported, please implement it first.".format(
                name=name)
        )
        return None

    transforms = get_transform(cfg.transforms)
    depth_transforms = get_transform(cfg.depth_transforms)
    target_transforms = get_transform(cfg.target_transforms)
    label_transforms = get_transform(cfg.label_transforms)
    common_transforms = get_common_transform(cfg.common_transforms)

    return ALL_DATASETS[name](
        **cfg.params,
        transforms=transforms,
        target_transforms=target_transforms,
        depth_transforms=depth_transforms,
        label_transforms=label_transforms,
        common_transforms=common_transforms
    )


def get_common_transform(transforms: DictConfig):
    transform_list = []
    if transforms is None:
        return None
    for name in transforms.keys():
        assert name in ALL_CUSTOM_TRANSFORM, (
            "{T_name} is not supported transform, please implement it and add it to "
            "ALL_TRANSFORM first.".format(T_name=name)
        )
        if transforms[name].params is not None:
            transform_list.append(
                ALL_CUSTOM_TRANSFORM[name](**transforms[name].params))
        else:
            transform_list.append(ALL_CUSTOM_TRANSFORM[name]())
    return CustomCompose(transform_list)


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
            transform_list.append(
                ALL_TRANSFORM[name](**transforms[name].params))
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
