from datasets import Iterator
import torch
import cv2
import torch.nn.functional as F
import os
import torch.nn as nn
from tqdm import tqdm

from utils.helper import Timer, Average_Meter
from torchvision.utils import save_image


class BaseRunner():
    def __init__(self, model, optimizer, losses, train_loader, val_loader, scheduler):
        self.optimizer = optimizer
        self.losses = losses
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.scheduler = scheduler
        self.trainer_timer = Timer()
        self.eval_timer = Timer()
        try:
            use_gpu = os.environ["CUDA_VISIBLE_DEVICES"]
        except KeyError:
            use_gpu = "0"
        self.the_number_of_gpu = len(use_gpu.split(","))

        if self.the_number_of_gpu > 1:
            self.model = nn.DataParallel(self.model)


class VAERunner(BaseRunner):
    def __init__(self, model, optimizer, losses, train_loader, val_loader, scheduler):
        super().__init__(model, optimizer, losses, train_loader, val_loader, scheduler)
        self.exist_status = ["train", "val", "test"]
        self.sample_dir = "samples/test"

    def train(self, cfg):
        train_meter = Average_Meter(list(self.losses.keys()) + ["total_loss"])

        for epoch in range(cfg.num_epoch):
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

                # Log and eval here.
                if (iteration + 1) % 10 == 0:
                    print("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}, KL Div: {:.4f}"
                          .format(epoch + 1, cfg.num_epoch, iteration + 1, len(self.train_loader),
                                  loss_dict["reconstruction_loss"].item(), loss_dict["kl_divergence_loss"].item()))

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
            loss_dict[item[0]] = tmp_loss
            total_loss += self.losses[item[0]]["weight"] * tmp_loss
