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
from utils import misc, lr_sched
import time
import datetime
import json
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
from torch.utils.tensorboard import SummaryWriter


class BaseRunner():
    def __init__(self, model, optimizer, losses, scheduler, train_loader, val_loader):
        self.optimizer = optimizer
        self.losses = losses
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.scheduler = scheduler
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


class VAERunner(BaseRunner):
    def __init__(self, model, optimizer, losses, scheduler, train_loader, val_loader):
        super().__init__(model, optimizer, losses, scheduler, train_loader, val_loader)
        self.exist_status = ["train", "val", "test"]
        self.sample_dir = "samples/test"

    def train(self, cfg):
        train_meter = Average_Meter(list(self.losses.keys()) + ["total_loss"])

        for epoch in range(cfg.num_epoch):
            self.model.train()
            for iteration, (x, _) in enumerate(self.train_loader):
                # Forward pass
                x = x.cuda().view(-1, cfg.model.params.image_size)
                x_reconst, mu, log_var = self.model(x)

                # Calculate losses
                total_loss = torch.zeros(1).cuda()
                loss_dict = {}
                self._compute_loss(total_loss, loss_dict, x,
                                   x_reconst, mu, log_var)

                # Backprop and optimize
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                loss_dict["total_loss"] = total_loss.item()
                train_meter.add(loss_dict)

                mean_loss = train_meter.get(loss_dict.keys())
                # Log and eval here.
                if (iteration + 1) % 10 == 0:
                    print("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}, KL Div: {:.4f}"
                          .format(epoch + 1, cfg.num_epoch, iteration + 1, len(self.train_loader),
                                  mean_loss["reconstruction_loss"], mean_loss["kl_divergence_loss"]))

            self.model.eval()
            with torch.no_grad():
                # Save the sampled images
                z = torch.randn(cfg.batch_size,
                                cfg.model.params.z_dim).cuda()
                out = self.model.decode(z).view(-1, 1, 28, 28)
                save_image(out, os.path.join(
                    self.sample_dir, 'sampled-{}.png'.format(epoch+1)))

                # Save the reconstructed images
                out, _, _ = self.model(x)
                x_concat = torch.cat(
                    [x.view(-1, 1, 28, 28), out.view(-1, 1, 28, 28)], dim=3)
                save_image(x_concat, os.path.join(
                    self.sample_dir, 'reconst-{}.png'.format(epoch+1)))

    def _compute_loss(self, total_loss, loss_dict, x, x_reconst, mu, log_var):
        for item in self.losses.items():
            # item: (key, value)
            # key: the illustrative name of the loss
            # value: includes loss function type and its parameters
            if item[0] == "reconstruction_loss":
                tmp_loss = item[1]["loss_func"](x_reconst, x)

            if item[0] == "kl_divergence_loss":
                tmp_loss = item[1]["loss_func"](log_var, mu)
            loss_dict[item[0]] = tmp_loss.item()
            total_loss += self.losses[item[0]]["weight"] * tmp_loss


class MAERunner(BaseRunner):
    def __init__(self, model, optimizer, losses, scheduler, train_loader, val_loader):
        super().__init__(model, optimizer, losses, scheduler, train_loader, val_loader)
        self.loss_scaler = NativeScaler()

    @staticmethod
    def train_one_epoch(model: torch.nn.Module,
                        data_loader: Iterable, optimizer: torch.optim.Optimizer,
                        device: torch.device, epoch: int, loss_scaler,
                        log_writer=None,
                        cfg=None):
        model.train(True)
        metric_logger = misc.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', misc.SmoothedValue(
            window_size=1, fmt='{value:.6f}'))
        header = 'Epoch: [{}]'.format(epoch)
        print_freq = 20
        
        accum_iter = cfg.accum_iter

        optimizer.zero_grad()

        if log_writer is not None:
            print('log_dir: {}'.format(log_writer.log_dir))

        for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            # we use a per iteration (instead of per epoch) lr scheduler
            if data_iter_step % accum_iter == 0:
                lr_sched.adjust_learning_rate(
                    optimizer, data_iter_step / len(data_loader) + epoch, cfg)
            if isinstance(samples, dict):
                for key in samples.keys():
                    samples[key] = samples[key].to(device, non_blocking=True)
            else:
                samples = samples.to(device, non_blocking=True)
            
            with torch.cuda.amp.autocast():
                # loss, _, _ = model(samples, mask_ratio=cfg.mask_ratio)
                loss, _, _ = model(samples)

            loss_value = loss.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            loss /= accum_iter
            loss_scaler(loss, optimizer, parameters=model.parameters(),
                        update_grad=(data_iter_step + 1) % accum_iter == 0)
            if (data_iter_step + 1) % accum_iter == 0:
                optimizer.zero_grad()

            torch.cuda.synchronize()
            
            metric_logger.update(loss=loss_value)

            lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(lr=lr)

            loss_value_reduce = misc.all_reduce_mean(loss_value)
            if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
                """ We use epoch_1000x as the x-axis in tensorboard.
                This calibrates different curves when batch size changes.
                """
                epoch_1000x = int(
                    (data_iter_step / len(data_loader) + epoch) * 1000)
                log_writer.add_scalar(
                    'train_loss', loss_value_reduce, epoch_1000x)
                log_writer.add_scalar('lr', lr, epoch_1000x)

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    def train(self, cfg):
        os.makedirs(cfg.log_dir, exist_ok=True)
        if cfg.log_dir is not None:
            self.log_writer = SummaryWriter(log_dir=cfg.log_dir)
        self.model.to(cfg.device)
        self.model_without_ddp = self.model
        if cfg.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[cfg.gpu], find_unused_parameters=True)
            self.model_without_ddp = self.model.module
        
        start_epoch = 0    
        if cfg.resume is not None:
            start_epoch = misc.load_model_to_resume(cfg, self.model, optimizer=self.optimizer, loss_scaler=self.loss_scaler)
        
        self.model.train()
        start_time = time.time()
        for epoch in range(start_epoch, cfg.num_epochs):
            if cfg.distributed:
                self.train_loader.sampler.set_epoch(epoch)
            train_stats = self.train_one_epoch(
                self.model, self.train_loader,
                self.optimizer, cfg.device, epoch, self.loss_scaler,
                log_writer=self.log_writer,
                cfg=cfg
            )
            if not os.path.exists(cfg.output_dir):
                os.makedirs(cfg.output_dir)
            if cfg.output_dir and (epoch % cfg.saving_interval == 0 or epoch + 1 == cfg.num_epochs) and epoch > cfg.start_saving_epoch:
                misc.save_model(
                    args=cfg, model=self.model, model_without_ddp=self.model_without_ddp, optimizer=self.optimizer,
                    loss_scaler=self.loss_scaler, epoch=epoch)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch, }

            if cfg.output_dir and misc.is_main_process():
                if self.log_writer is not None:
                    self.log_writer.flush()
                with open(os.path.join(cfg.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))
        
