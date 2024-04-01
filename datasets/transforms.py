import torchvision.transforms as T
import torch.nn as nn
from omegaconf.dictconfig import DictConfig

ALL_TRANSFORM = {"resize": T.Resize, "to_tensor": T.ToTensor}


def get_transform(transforms: DictConfig):
    transform_list = []
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


class CustomTransform(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass
