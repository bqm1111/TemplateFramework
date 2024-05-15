import argparse
from omegaconf import OmegaConf
from omegaconf.errors import ConfigKeyError
from torch.utils.data import DataLoader
from datasets import get_dataset
from losses import get_losses
from engine import get_model, get_opt_params, get_optimizer, get_scheduler, get_runner
from utils.init_func import group_weight
from timm.optim import optim_factory
import torch.nn as nn
parser = argparse.ArgumentParser()
parser.add_argument("--config", help="Path to config file")

if __name__ == "__main__":
    args = parser.parse_args()
    config = OmegaConf.load(args.config)

    train_cfg = config.train
    if "val" in config:
        val_cfg = config.val
    else:
        val_cfg = None
    if "test" in config:
        test_cfg = config.test
    else:
        test_cfg = None

    train_dataset = get_dataset(train_cfg.dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        drop_last=train_cfg.drop_last,
    )
    if val_cfg is not None:
        val_dataset = get_dataset(val_cfg.dataset)
    else:
        val_dataset = None

    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=val_cfg.batch_size,
            shuffle=False,
            num_workers=val_cfg.num_workers,
            drop_last=val_cfg.drop_last,
        )
    else:
        val_loader = None
    losses = get_losses(losses=train_cfg.losses)

    # according the model name to get the adapted model
    model = get_model(model_name=config.model.name,
                      **config.model.params)
    # TODO: Unify interface for MAE and SemSeg training

    if train_cfg.experiment_name == "mae":
        opt_params = optim_factory.param_groups_weight_decay(
            model, train_cfg.weight_decay)
    elif train_cfg.experiment_name == "semseg":
        opt_params = group_weight(model, config.model.params.norm_layer, train_cfg.base_lr)

    optimizer = get_optimizer(
        opt_name=train_cfg.opt_name,
        params=opt_params,
        lr=train_cfg.opt_params.lr_default,
        momentum=train_cfg.opt_params.momentum,
        weight_decay=train_cfg.opt_params.wd_default,
    )
    scheduler = get_scheduler(
        optimizer=optimizer, lr_scheduler=train_cfg.scheduler_name
    )

    runner = get_runner(train_cfg)(
        model, optimizer, losses, scheduler, train_loader, val_loader, train_cfg, val_cfg, test_cfg
    )

    # train_step
    runner.train()
    if test_cfg is not None and test_cfg.need_test:
        runner.test()
