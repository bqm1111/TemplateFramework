# copyright ziqi-jin
import torch
from models.vae import VAE
from models.vanilla_mae import (
    mae_vit_base_patch16,
    mae_vit_huge_patch14,
    mae_vit_large_patch16,
)
from models.swin_mae import swin_mae
from models.reimplement.my_swin_mae import my_swin_mae
from models.reimplement.dual_swin_mae import (
    dual_swinmae_t,
    dual_swinmae_s,
    dual_swinmae_b,
)
from .runner import BaseRunner, VAERunner, MAERunner

# from .optimizer import BaseOptimizer
from .scheduler import WarmupMultiStepLR

AVAI_SCH = [
    "single_step",
    "multi_step",
    "warmup_multi_step",
    "cosine",
    "linear",
    "constant",
]
AVAI_MODEL = {
    "vae": VAE,
    "mae_vit_base_patch16": mae_vit_base_patch16,
    "mae_vit_huge_patch14": mae_vit_huge_patch14,
    "mae_vit_large_patch16": mae_vit_large_patch16,
    "swin_mae": swin_mae,
    "my_swin_mae": my_swin_mae,
    "dual_swinmae_t": dual_swinmae_t,
    "dual_swinmae_b": dual_swinmae_b,
    "dual_swinmae_s": dual_swinmae_s,
}
# AVAI_OPT = {'base_opt': BaseOptimizer, 'sgd': torch.optim.SGD, 'adam': torch.optim.Adam}
AVAI_OPT = {
    "sgd": torch.optim.SGD,
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
}
AVAI_RUNNER = {"base_runner": BaseRunner, "vae": VAERunner, "mae": MAERunner}


def get_model(model_name, **kwargs):
    if model_name not in AVAI_MODEL:
        print("not supported model name, please implement it first.")
    return AVAI_MODEL[model_name](**kwargs).cuda()


def get_optimizer(opt_name, **kwargs):
    if opt_name not in AVAI_OPT:
        print("not supported optimizer name, please implement it first.")
    return AVAI_OPT[opt_name](**{k: v for k, v in kwargs.items() if v is not None})


def get_runner(runner_name):
    if runner_name not in AVAI_RUNNER:
        print("not supported runner name, please implement it first.")
    return AVAI_RUNNER[runner_name]


def get_scheduler(
    optimizer,
    lr_scheduler="single_step",
    stepsize=1,
    gamma=0.1,
    warmup_factor=0.01,
    warmup_steps=10,
    max_epoch=1,
    n_epochs_init=50,
    n_epochs_decay=50,
):
    """A function wrapper for building a learning rate scheduler.
    Args:
        optimizer (Optimizer): an Optimizer.
        lr_scheduler (str, optional): learning rate scheduler method. Default is
            single_step.
        stepsize (int or list, optional): step size to decay learning rate.
            When ``lr_scheduler`` is "single_step", ``stepsize`` should be an integer.
            When ``lr_scheduler`` is "multi_step", ``stepsize`` is a list. Default is 1.
        gamma (float, optional): decay rate. Default is 0.1.
        max_epoch (int, optional): maximum epoch (for cosine annealing). Default is 1.
    Examples::
        >>> # Decay learning rate by every 20 epochs.
        >>> scheduler = get_scheduler(
        >>>     optimizer, lr_scheduler='single_step', stepsize=20
        >>> )
        >>> # Decay learning rate at 30, 50 and 55 epochs.
        >>> scheduler = get_scheduler(
        >>>     optimizer, lr_scheduler='multi_step', stepsize=[30, 50, 55]
        >>> )
    """
    if lr_scheduler not in AVAI_SCH:
        raise ValueError(
            "Unsupported scheduler: {}. Must be one of {}".format(
                lr_scheduler, AVAI_SCH
            )
        )

    if lr_scheduler == "single_step":
        if isinstance(stepsize, list):
            stepsize = stepsize[-1]

        if not isinstance(stepsize, int):
            raise TypeError(
                "For single_step lr_scheduler, stepsize must "
                "be an integer, but got {}".format(type(stepsize))
            )

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=stepsize, gamma=gamma
        )

    elif lr_scheduler == "multi_step":
        if not isinstance(stepsize, list):
            raise TypeError(
                "For multi_step lr_scheduler, stepsize must "
                "be a list, but got {}".format(type(stepsize))
            )

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=stepsize, gamma=gamma
        )

    elif lr_scheduler == "warmup_multi_step":
        if not isinstance(stepsize, list):
            raise TypeError(
                "For warmup multi_step lr_scheduler, stepsize must "
                "be a list, but got {}".format(type(stepsize))
            )

        scheduler = WarmupMultiStepLR(
            optimizer,
            milestones=stepsize,
            gamma=gamma,
            warmup_factor=warmup_factor,
            warmup_iters=warmup_steps,
        )

    elif lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, int(max_epoch)
        )

    elif lr_scheduler == "linear":

        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - n_epochs_init) / float(n_epochs_decay + 1)
            return lr_l

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif lr_scheduler == "constant":
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=warmup_factor)
    return scheduler


def get_opt_params(model, lr_list, group_keys, wd_list):
    """

    :param model: model
    :param lr_list: list, contain the lr for each params group
    :param wd_list: list, contain the weight decay for each params group
    :param group_keys: list of list, according to the sub list to divide params to different groups
    :return: list of dict
    """
    for name, value in model.named_parameters():
        print("parameter named: ", name)

    if lr_list is not None:
        assert len(lr_list) == len(
            group_keys
        ), "lr_list should has the same length as group_keys"
        assert len(lr_list) == len(
            wd_list
        ), "lr_list should has the same length as wd_list"
        params_group = [[] for _ in range(len(lr_list))]
        for name, value in model.named_parameters():
            print("parameter named: ", name)
            for index, g_keys in enumerate(group_keys):
                for g_key in g_keys:
                    if g_key in name:
                        params_group[index].append(value)
        return [
            {"params": params_group[i], "lr": lr_list[i], "weight_decay": wd_list[i]}
            for i in range(len(lr_list))
        ]
