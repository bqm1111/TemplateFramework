import argparse

import torch
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
from models import vanilla_mae, swin_mae
from utils.logger import get_root_logger

# define the utils
# mean: [0.4939, 0.4259, 0.4036]
# std: [0.2896, 0.2954, 0.3072]


parser = argparse.ArgumentParser()
parser.add_argument("--use_norm", action="store_true")
parser.add_argument("--rgb", action="store_true")
parser.add_argument("--epoch", type=int, default=1040)
logger = get_root_logger()

if __name__ == "__main__":
    args = parser.parse_args()

    # # rgb mean and std
    if args.rgb:
        IMG_MEAN = np.array([0.4939, 0.4259, 0.4036])
        IMG_STD = np.array([0.2896, 0.2954, 0.3072])
    else:
    # depth mean and std
        IMG_MEAN = np.array([0.4132, 0.4132, 0.4132])
        IMG_STD = np.array([0.2703, 0.2703, 0.2703])

    def show_image(image, use_norm, title=""):
        # image is [H, W, 3]
        assert image.shape[2] == 3
        if use_norm:
            plt.imshow(torch.clip((image * IMG_STD + IMG_MEAN) * 255, 0, 255).int())
        else:
            plt.imshow(torch.clip(image * 255, 0, 255).int())

        plt.title(title, fontsize=16)
        plt.axis("off")
        return

    def prepare_model(chkpt_dir, arch="mae_vit_large_patch16"):
        # build model
        model = getattr(swin_mae, arch)()
        # load model
        checkpoint = torch.load(chkpt_dir, map_location="cpu")
        msg = model.load_state_dict(checkpoint["model"], strict=False)
        print(msg)
        return model

    def run_one_image(img, model, use_norm):
        x = torch.tensor(img)

        # make it a batch-like
        x = x.unsqueeze(dim=0)
        x = torch.einsum("nhwc->nchw", x)

        # run MAE
        # loss, y, mask = model(x.float(), mask_ratio=0.75)
        loss, y, mask = model(x.float())

        y = model.unpatchify(y)
        y = torch.einsum("nchw->nhwc", y).detach().cpu()

        # visualize the mask
        mask = mask.detach()
        print(f"Shape of mask = {mask.shape}")
        mask = mask.unsqueeze(-1).repeat(
            1, 1, model.patch_embed.patch_size**2 * 3
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
        show_image(x[0], use_norm, "original")

        plt.subplot(1, 4, 2)
        show_image(im_masked[0], use_norm, "masked")

        plt.subplot(1, 4, 3)
        show_image(y[0], use_norm, "reconstruction")

        plt.subplot(1, 4, 4)
        show_image(im_paste[0], use_norm, "reconstruction + visible")

        plt.show()

    # load an image
    if args.rgb:
        filename = "data/sunrgbd_trainval/image/000013.jpg"
    else:
        filename = "data/sunrgbd_trainval/depth/001013.png"

    img = Image.open(filename)
    if args.rgb:
        img = img.resize((224, 224))
        img = np.array(img) / 255.0
    else:
        img = np.array(img)
        img = Image.fromarray((img / np.max(img) * 255.0).astype(np.uint8))
        img = img.resize((224, 224))
        img = np.array(img) / 255.0
        img = np.stack((img,) * 3, axis=-1)

    assert img.shape == (224, 224, 3)

    # normalize by ImageNet mean and std
    if args.use_norm:
        print("Using normalization")
        img = img - IMG_MEAN
        img = img / IMG_STD

    plt.rcParams["figure.figsize"] = [5, 5]
    # show_image(torch.tensor(img))
    if args.use_norm:
        chkpt_dir = "output_dir/rgb_normalized/checkpoint-" + str(args.epoch) + ".pth"
    else:
        if args.rgb:
            chkpt_dir = "output_dir/no_normalize/checkpoint-" + str(args.epoch) + ".pth"
        else:
            chkpt_dir = "output_dir/depth_unnormalized/checkpoint-" + str(args.epoch) + ".pth"

    model_mae = prepare_model(chkpt_dir, "swin_mae")
    print("Model loaded.")
    torch.manual_seed(2)
    print("MAE with pixel reconstruction:")
    run_one_image(img, model_mae, args.use_norm)
