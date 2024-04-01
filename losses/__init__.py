import torch.nn as nn
from .losses import CustomLoss, KLDivergenceLoss, BinaryCrossEntropy
import torch.nn.functional as F
AVAI_LOSS = {'ce': nn.CrossEntropyLoss, 'multi_label_soft_margin': nn.MultiLabelSoftMarginLoss,
             'test_custom': CustomLoss, 'mse': nn.MSELoss,
             'binary_cross_entropy': nn.BCELoss,
             'KLDivLoss': KLDivergenceLoss}


def get_losses(losses):
    loss_dict = {}
    for name in losses:
        assert losses[name]['type'] in AVAI_LOSS, print(
            '{name} is not supported, please implement it first.'.format(name=losses[name]['type']))
        if losses[name].params is not None:
            loss_dict[name] = {'loss_func': AVAI_LOSS[losses[name]['type']](
                **losses[name].params), 'weight': losses[name]['weight']}
        else:
            loss_dict[name] = {'loss_func': AVAI_LOSS[losses[name]
                                                      ['type']](), 'weight': losses[name]['weight']}
    return loss_dict
