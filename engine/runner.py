from datasets import Iterator
import torch
import cv2
import torch.nn.functional as F
import os
import torch.nn as nn
from tqdm import tqdm

from utils.helper import Timer, Average_Meter
from torchvision.utils import save_image
import timm
import timm.optim.optim_factory as optim_factory
import math
from utils.logger import get_root_logger
import sys
from typing import Iterable
from utils import misc, lr_sched, lr_policy
import time
import datetime
import json
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
from utils.misc import all_reduce_tensor, all_reduce_mean
from torch.utils.tensorboard import SummaryWriter
logger = get_root_logger()


class BaseRunner():
    def __init__(self, model, optimizer, losses, scheduler, train_loader, val_loader=None, train_cfg=None, val_cfg=None, test_cfg=None):
        self.optimizer = optimizer
        self.losses = losses
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.scheduler = scheduler
        self.train_cfg = train_cfg
        self.val_cfg = val_cfg
        self.test_cfg = test_cfg
        self.trainer_timer = Timer()
        self.eval_timer = Timer()
        self.logger = get_root_logger()

        try:
            use_gpu = os.environ["CUDA_VISIBLE_DEVICES"]
        except KeyError:
            use_gpu = "0"
        self.the_number_of_gpu = len(use_gpu.split(","))

        if self.the_number_of_gpu > 1:
            self.model = nn.DataParallel(self.model)

    def train_one_epoch(self):
        raise NotImplementedError

    def train(self):
        cfg = self.train_cfg
        self.device = cfg.device
        os.makedirs(cfg.log_dir, exist_ok=True)
        if cfg.log_dir is not None:
            self.log_writer = SummaryWriter(log_dir=cfg.log_dir)
        self.model.to(cfg.device)
        self.model_without_ddp = self.model
        if cfg.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[cfg.gpu], find_unused_parameters=True)
            self.model_without_ddp = self.model.module

        start_epoch = 1
        if cfg.resume is not None:
            start_epoch = misc.load_model_to_resume(
                cfg, self.model, optimizer=self.optimizer, loss_scaler=self.loss_scaler)

        self.model.train()
        start_time = time.time()
        if not os.path.exists(cfg.output_dir):
            os.makedirs(cfg.output_dir)

        for epoch in range(start_epoch, cfg.num_epochs + 1):
            self.epoch = epoch
            if cfg.distributed:
                self.train_loader.sampler.set_epoch(epoch)

            self.train_one_epoch()

            if cfg.output_dir and (epoch % cfg.saving_interval == 0 or epoch == cfg.num_epochs) and epoch >= cfg.start_saving_epoch:
                misc.save_model(
                    args=cfg, model=self.model, model_without_ddp=self.model_without_ddp, optimizer=self.optimizer,
                    loss_scaler=self.loss_scaler, epoch=epoch)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info('Training time {}'.format(total_time_str))


class MAERunner(BaseRunner):
    def __init__(self, model, optimizer, losses, scheduler, train_loader, val_loader=None, train_cfg=None, val_cfg=None, test_cfg=None):
        super().__init__(model, optimizer, losses, scheduler,
                         train_loader, val_loader, train_cfg, val_cfg, test_cfg)
        self.loss_scaler = NativeScaler()
        self.train_cfg = train_cfg

    def train_one_epoch(self):
        cfg = self.train_cfg
        self.model.train(True)
        metric_logger = misc.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', misc.SmoothedValue(
            window_size=1, fmt='{value:.6f}'))
        header = 'Epoch: [{}]'.format(self.epoch)
        print_freq = 20

        accum_iter = cfg.accum_iter

        self.optimizer.zero_grad()

        if self.log_writer is not None:
            logger.info('log_dir: {}'.format(self.log_writer.log_dir))

        for data_iter_step, samples in enumerate(metric_logger.log_every(self.train_loader, print_freq, header)):
            # we use a per iteration (instead of per epoch) lr scheduler
            if data_iter_step % accum_iter == 0:
                lr_sched.adjust_learning_rate(
                    self.optimizer, data_iter_step / len(self.train_loader) + self.epoch, cfg)
            if isinstance(samples, dict):
                for key in samples.keys():
                    samples[key] = samples[key].to(
                        self.device, non_blocking=True)
            else:
                samples = samples.to(self.device, non_blocking=True)

            with torch.cuda.amp.autocast():
                loss, _, _ = self.model(samples)

            loss_value = loss.item()

            if not math.isfinite(loss_value):
                logger.error(
                    "Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            loss /= accum_iter
            self.loss_scaler(loss, self.optimizer, parameters=self.model.parameters(),
                             update_grad=(data_iter_step + 1) % accum_iter == 0)
            if (data_iter_step + 1) % accum_iter == 0:
                self.optimizer.zero_grad()

            torch.cuda.synchronize()

            metric_logger.update(loss=loss_value)

            lr = self.optimizer.param_groups[0]["lr"]
            metric_logger.update(lr=lr)

            loss_value_reduce = misc.all_reduce_mean(loss_value)
            if self.log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
                """ We use epoch_1000x as the x-axis in tensorboard.
                This calibrates different curves when batch size changes.
                """
                epoch_1000x = int(
                    (data_iter_step / len(self.train_loader) + self.epoch) * 1000)
                self.log_writer.add_scalar(
                    'train_loss', loss_value_reduce, epoch_1000x)
                self.log_writer.add_scalar('lr', lr, epoch_1000x)

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        train_stats = {k: meter.global_avg for k,
                       meter in metric_logger.meters.items()}
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': self.epoch, }

        if cfg.output_dir and misc.is_main_process():
            if self.log_writer is not None:
                self.log_writer.flush()
            with open(os.path.join(cfg.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")


class SemSegRunner(BaseRunner):
    def __init__(self, model, optimizer, losses, scheduler, train_loader, val_loader=None, train_cfg=None, val_cfg=None, test_cfg=None):
        super().__init__(model, optimizer, losses, scheduler,
                         train_loader, val_loader, train_cfg, val_cfg, test_cfg)
        self.train_cfg = train_cfg
        self.loss_scaler = None
        niters_per_epoch = len(self.train_loader)
        total_iteration = self.train_cfg.num_epochs * niters_per_epoch
        self.scheduler = lr_policy.WarmUpPolyLR(
            train_cfg.base_lr, train_cfg.lr_power, total_iteration, niters_per_epoch * train_cfg.warm_up_epoch)

    def train_one_epoch(self):
        cfg = self.train_cfg
        self.model.train(True)
        sum_loss = 0
        for data_iter_step, samples in enumerate(self.train_loader):
            rgb = samples["rgb"].cuda(non_blocking=True)
            depth = samples["depth"].cuda(non_blocking=True)
            label = samples["seglabel"].cuda(non_blocking=True)

            aux_rate = 0.2
            loss = self.model(rgb, depth, label)
            if cfg.distributed:
                reduce_loss = all_reduce_tensor(
                    loss, world_size=cfg.world_size)

            self.optimizer.zero_grad()
            self.losses.backward()
            self.optimizer.step()

            # TODO: Make this lr scheduler more efficient
            current_step = (self.epoch - 1) * \
                len(self.train_loader) + data_iter_step
            lr = self.scheduler.get_lr(current_step)

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            if cfg.distributed:
                sum_loss += reduce_loss.item()
                print_str = 'Epoch {}/{}'.format(self.epoch, cfg.num_epochs) \
                    + ' Iter {}/{}:'.format(data_iter_step + 1, len(self.train_loader)) \
                    + ' lr=%.4e' % lr \
                    + ' loss=%.4f total_loss=%.4f' % (reduce_loss.item(), (sum_loss / (data_iter_step + 1)))
            else:
                sum_loss += loss
                print_str = 'Epoch {}/{}'.format(self.epoch, cfg.num_epochs) \
                    + ' Iter {}/{}:'.format(data_iter_step + 1, len(self.train_loader)) \
                    + ' lr=%.4e' % lr \
                    + ' loss=%.4f total_loss=%.4f' % (loss, (sum_loss / (data_iter_step + 1)))

            del loss
        if (cfg.distributed and (cfg.local_rank == 0)) or (not cfg.distributed):
            self.log_writer.add_scalar(
                'train_loss', sum_loss / len(self.train_loader), self.epoch)
