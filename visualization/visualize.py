import sys
import os
import requests

import torch
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
from models import vanilla_mae

# define the utils
# mean: [0.4939, 0.4259, 0.4036]
# std: [0.2896, 0.2954, 0.3072]

img_mean = np.array([0.4939, 0.4259, 0.4036])
img_std = np.array([0.2896, 0.2954, 0.3072])


def show_image(image, title=""):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    plt.imshow(torch.clip((image * img_std + img_mean) * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis("off")
    return


def prepare_model(chkpt_dir, arch="mae_vit_large_patch16"):
    # build model
    model = getattr(vanilla_mae, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location="cpu")
    msg = model.load_state_dict(checkpoint["model"], strict=False)
    print(msg)
    return model


def run_one_image(img, model):
    x = torch.tensor(img)

    # make it a batch-like
    x = x.unsqueeze(dim=0)
    x = torch.einsum("nhwc->nchw", x)

    # run MAE
    loss, y, mask = model(x.float(), mask_ratio=0.75)
    y = model.unpatchify(y)
    y = torch.einsum("nchw->nhwc", y).detach().cpu()

    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(
        1, 1, model.patch_embed.patch_size[0] ** 2 * 3
    )  # (N, H*W, p*p*3)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum("nchw->nhwc", mask).detach().cpu()

    x = torch.einsum("nchw->nhwc", x)

    # masked image
    im_masked = x * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask

    # make the plt figure larger
    plt.rcParams["figure.figsize"] = [24, 24]

    plt.subplot(1, 4, 1)
    show_image(x[0], "original")

    plt.subplot(1, 4, 2)
    show_image(im_masked[0], "masked")

    plt.subplot(1, 4, 3)
    show_image(y[0], "reconstruction")

    plt.subplot(1, 4, 4)
    show_image(im_paste[0], "reconstruction + visible")

    plt.show()


# load an image
filename = "../../datasets/NYUDepthv2/RGB/13.jpg"
img = Image.open(filename)
img = img.resize((224, 224))
img = np.array(img) / 255.0

assert img.shape == (224, 224, 3)

# normalize by ImageNet mean and std
img = img - img_mean
img = img / img_std

plt.rcParams["figure.figsize"] = [5, 5]
# show_image(torch.tensor(img))
chkpt_dir = "output_dir/baseline/checkpoint-799.pth"
model_mae = prepare_model(chkpt_dir, "mae_vit_large_patch16")
print("Model loaded.")
torch.manual_seed(2)
print("MAE with pixel reconstruction:")
run_one_image(img, model_mae)
