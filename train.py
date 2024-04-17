import argparse
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from datasets import get_dataset
from losses import get_losses
from engine import get_model, get_opt_params, get_optimizer, get_scheduler, get_runner
import cv2

parser = argparse.ArgumentParser()


if __name__ == "__main__":
    config = OmegaConf.load("config/mae.yaml")

    train_cfg = config.train
    val_cfg = config.val
    test_cfg = config.test

    train_dataset = get_dataset(train_cfg.dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        drop_last=train_cfg.drop_last,
    )

    val_dataset = get_dataset(val_cfg.dataset)

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
    model = get_model(model_name=train_cfg.model.name,
                      **train_cfg.model.params)
    opt_params = get_opt_params(
        model,
        lr_list=train_cfg.opt_params.lr_list,
        group_keys=None,
        wd_list=train_cfg.opt_params.wd_list,
    )
    optimizer = get_optimizer(
        opt_name=train_cfg.opt_name,
        params=model.parameters(),
        lr=train_cfg.opt_params.lr_default,
        momentum=train_cfg.opt_params.momentum,
        weight_decay=train_cfg.opt_params.wd_default,
    )
    scheduler = get_scheduler(
        optimizer=optimizer, lr_scheduler=train_cfg.scheduler_name
    )
    runner = get_runner(train_cfg.runner_name)(
        model, optimizer, losses, scheduler, train_loader, val_loader
    )

    # train_step
    runner.train(train_cfg)
    if test_cfg.need_test:
        runner.test(test_cfg)
