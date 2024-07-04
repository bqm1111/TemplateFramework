from numpy import save
from datasets import Iterator
import torch
import cv2
import torch.nn.functional as F
import os
import torch.nn as nn
from tqdm import tqdm

from engine.evaluator import Evaluator
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
from utils.helper import link_file
logger = get_root_logger()


class BaseRunner():
    def __init__(self, config, model, optimizer, losses, scheduler, train_loader, val_loader=None):
        self.optimizer = optimizer
        self.losses = losses
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.scheduler = scheduler
        self.cfg = config
        self.train_cfg = config.train
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
        train_cfg = self.train_cfg
        self.device = train_cfg.device
        os.makedirs(train_cfg.log_dir, exist_ok=True)
        if train_cfg.log_dir is not None:
            self.log_writer = SummaryWriter(log_dir=os.path.join(
                train_cfg.log_dir, self.cfg.experiment_type, self.cfg.experiment_dataset, self.cfg.experiment_name))
        
        self.model.to(self.device)
        self.model_without_ddp = self.model
        if train_cfg.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[train_cfg.device], find_unused_parameters=False)
            self.model_without_ddp = self.model.module

        start_epoch = 1
        if train_cfg.resume is not None:
            start_epoch = misc.load_model_to_resume(
                train_cfg, self.model, optimizer=self.optimizer, loss_scaler=self.loss_scaler)

        self.model.train()
        start_time = time.time()
        output_dir = os.path.join(
            train_cfg.output_dir, self.cfg.experiment_type, self.cfg.experiment_dataset, self.cfg.experiment_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        best_iou = 0
        for epoch in range(start_epoch, train_cfg.num_epochs + 1):
            self.epoch = epoch
            if train_cfg.distributed:
                self.train_loader.sampler.set_epoch(epoch)

            self.train_one_epoch()

            if train_cfg.output_dir and (epoch % train_cfg.saving_interval == 0 or epoch == train_cfg.num_epochs) and \
                    epoch >= train_cfg.start_saving_epoch:
                model_save_path = misc.save_model(
                    args=train_cfg, output_dir=output_dir, model=self.model, model_without_ddp=self.model_without_ddp, optimizer=self.optimizer,
                    loss_scaler=self.loss_scaler, epoch=epoch)
                if self.cfg.experiment_type == "semseg":
                    if misc.is_main_process():
                        self.model.eval()
                        save_path = model_save_path[0]
                        segmentor = Evaluator(
                            self.cfg, self.model_without_ddp, show=False)
                        meanIOU = segmentor.run(save_path, need_load=False)
                        if meanIOU > best_iou:
                            best_iou = meanIOU
                            best_path = os.path.join(
                                os.path.dirname(save_path), "best" + ".pth")
                            link_file(save_path, best_path)

                        self.model.train()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info('Training time {}'.format(total_time_str))


class MAERunner(BaseRunner):
    def __init__(self, config, model, optimizer, losses, scheduler, train_loader, val_loader=None):
        super().__init__(config, model, optimizer, losses, scheduler,
                         train_loader, val_loader)
        self.loss_scaler = NativeScaler()

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
    def __init__(self, config, model, optimizer, losses, scheduler, train_loader, val_loader=None):
        super().__init__(config, model, optimizer, losses, scheduler,
                         train_loader, val_loader)
        self.loss_scaler = None
        niters_per_epoch = len(self.train_loader)
        total_iteration = self.train_cfg.num_epochs * niters_per_epoch
        self.scheduler = lr_policy.WarmUpPolyLR(
            self.train_cfg.lr, self.train_cfg.lr_power, total_iteration, niters_per_epoch * self.train_cfg.warmup_epoch)

    def train_one_epoch(self):
        cfg = self.train_cfg
        sum_loss = 0
        for data_iter_step, samples in enumerate(self.train_loader):

            rgb = samples["rgb"].to(cfg.device)
            depth = samples["depth"].to(cfg.device)
            label = samples["label"].to(cfg.device)


            loss = self.model(rgb, depth, label)
            if cfg.distributed:
                reduce_loss = all_reduce_tensor(
                    loss, world_size=cfg.world_size)

            self.optimizer.zero_grad()
            loss.backward()
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
            logger.info(print_str)

        if (cfg.distributed and (misc.is_main_process())) or (not cfg.distributed):
            self.log_writer.add_scalar(
                'train_loss', sum_loss / len(self.train_loader), self.epoch)
            self.log_writer.add_scalar('lr', lr, self.epoch)
        
