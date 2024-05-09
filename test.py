from torchvision.transforms import ToTensor
from torchvision import datasets
from torch import nn
import torch
from omegaconf import OmegaConf

from datasets.datasets import DepthDataset
from torch.utils.data import DataLoader
import os
import cv2
from losses import get_losses
import numpy as np
from copy import deepcopy
from utils.logger import get_root_logger
import torch.nn as nn
from timm.optim import optim_factory

logger = get_root_logger()
config = OmegaConf.load("config/config.yaml")


training_data = datasets.FashionMNIST(
    root="data", train=True, download=True, transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data", train=False, download=True, transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


if __name__ == "__main__":
    # model = NeuralNetwork()
    # learning_rate = 1e-3
    # batch_size = 64
    # epochs = 5

    # loss_fn = nn.CrossEntropyLoss()
    # param_groups = optim_factory.param_groups_weight_decay(model, 1)
    # optimizer = torch.optim.SGD(param_groups, lr=learning_rate)
    # for name in model.parameters():
    #     print(name)
    # print(param_groups[0].keys())
    # for param in optimizer.param_groups:
    #     if param.keys() == "params":
    #         print(param.keys())

    # epochs = 10
    # for t in range(epochs):
    #     print(f"Epoch {t+1}\n-------------------------------")
    #     train_loop(train_dataloader, model, loss_fn, optimizer)
    #     test_loop(test_dataloader, model, loss_fn)
    # print("Done!")
    def func(d):
     d['a'] = 10
     d['b'] = 20
 
    d = {'a': 1, 'b': 2}
    func(d)
    print(d)

