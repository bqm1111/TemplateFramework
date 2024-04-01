from .transforms import get_transform
from .datasets import DepthDataset, MNIST

ALL_DATASETS = {
    # "nyuv2": NYUv2dataset,
    # "sunrgbd": SunRGBDDataset,
    "mnist": MNIST
}


def get_dataset(cfg):
    name = cfg.name
    assert name in ALL_DATASETS, print(
        "{name} is not supported, please implement it first.".format(name=name)
    )

    transform = get_transform(cfg.transforms)
    target_transform = get_transform(cfg.target_transforms)

    return ALL_DATASETS[name](**cfg.params, transform=transform, target_transform=None)


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
