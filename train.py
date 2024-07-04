import argparse
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from datasets import get_dataset
from losses import get_losses
from engine import get_model, get_opt_params, get_optimizer, get_scheduler, get_runner
from utils.init_func import group_weight
from timm.optim import optim_factory
import torch
import utils.misc as misc
import os
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

parser = argparse.ArgumentParser()
parser.add_argument("--config", help="Path to config file")


def init_distributed_process(rank, size, backend='nccl'):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend=backend, rank=rank, world_size=size)


def main(rank, world_size, config):
    train_cfg = config.train
    train_dataset = get_dataset(train_cfg.dataset)

    if train_cfg.distributed:
        init_distributed_process(rank, world_size)
        sampler_train = DistributedSampler(train_dataset)
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_cfg.batch_size,
            shuffle=False,
            sampler=sampler_train,
            num_workers=train_cfg.num_workers,
            drop_last=train_cfg.drop_last,
        )
        train_cfg.device = rank
        torch.cuda.set_device(rank)

    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_cfg.batch_size,
            shuffle=True,
            num_workers=train_cfg.num_workers,
            drop_last=train_cfg.drop_last,
        )
    losses = get_losses(losses=train_cfg.losses)

    # according the model name to get the adapted model
    model = get_model(model_name=config.model.name,
                      **config.model.params)
    # TODO: Unify interface for MAE and SemSeg training

    if config.experiment_type == "mae":
        opt_params = optim_factory.param_groups_weight_decay(
            model, train_cfg.weight_decay)
    elif config.experiment_type == "semseg":
        opt_params = group_weight(
            model, config.model.params.norm_layer, train_cfg.lr)

    optimizer = get_optimizer(
        opt_name=train_cfg.opt_name,
        params=opt_params,
        lr=train_cfg.opt_params.lr_default,
        momentum=train_cfg.opt_params.momentum,
        weight_decay=train_cfg.opt_params.wd_default
    )
    scheduler = get_scheduler(
        optimizer=optimizer, lr_scheduler=train_cfg.scheduler_name
    )

    runner = get_runner(config.experiment_type)(config, model, optimizer,
                                                losses, scheduler, train_loader, val_loader=None)

    # train_step
    runner.train()


if __name__ == "__main__":
    import torch.multiprocessing as mp
    args = parser.parse_args()
    torch.manual_seed(1234)
    config = OmegaConf.load(args.config)
    world_size = torch.cuda.device_count()

    if config.train.distributed:
        mp.spawn(main, args=(world_size, config), nprocs=world_size)
    else:
        main(rank=0, world_size=1, config=config)
