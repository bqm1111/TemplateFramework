import argparse
import torch
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
from models.backbone import dual_swin_mae
from utils.logger import get_root_logger
import os

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_path", type=str, help="Path to checkpoint")
parser.add_argument(
    "--use_norm", action="store_true", help="reconstruct normalized pixel if true"
)
parser.add_argument("--img", type=int, help="Ordinal number of an image")
parser.add_argument("--epoch", type=int, default=1040, help="Saved epoch used")
logger = get_root_logger()

if __name__ == "__main__":
    args = parser.parse_args()
    NORM_RGB = {
        "mean": np.array([0.4939, 0.4259, 0.4036]),
        "std": np.array([0.2896, 0.2954, 0.3072]),
    }
    NORM_DEPTH = {
        "mean": np.array([0.4132, 0.4132, 0.4132]),
        "std": np.array([0.2703, 0.2703, 0.2703]),
    }
    NORM_DEPTH_ANYTHING = {
        "mean": np.array([0.4975, 0.4975, 0.4975]),
        "std": np.array([0.2218, 0.2218, 0.2218]),
    }

    def show_image(image, use_norm, norm, title=""):
        # image is [H, W, 3]
        assert image.shape[2] == 3
        if use_norm:
            plt.imshow(
                torch.clip((image * norm["std"] + norm["mean"]) * 255, 0, 255).int()
            )
        else:
            plt.imshow(torch.clip(image * 255, 0, 255).int())

        plt.title(title, fontsize=16)
        plt.axis("off")
        return

    def prepare_model(chkpt_dir, arch="mae_vit_large_patch16"):
        # build model
        model = getattr(dual_swin_mae, arch)()
        # load model
        checkpoint = torch.load(chkpt_dir, map_location="cpu")
        msg = model.load_state_dict(checkpoint["model"], strict=False)
        print(msg)
        return model

    def run_model(img, model, use_norm):
        x = img["rgb"]
        x_d = img["depth"]
        depth_anything = img["depth_anything"]
        x = torch.tensor(x)
        x = x.unsqueeze(dim=0)
        x = torch.einsum("nhwc->nchw", x)

        x_d = torch.tensor(x_d)
        x_d = x_d.unsqueeze(dim=0)
        x_d = torch.einsum("nhwc->nchw", x_d)
        
        depth_anything = torch.tensor(depth_anything)
        depth_anything = depth_anything.unsqueeze(dim=0)
        depth_anything = torch.einsum("nhwc->nchw", depth_anything)

        img = {"rgb": x.float(), "depth": x_d.float(), "depth_anything": depth_anything}
        # run MAE
        # loss, y, mask = model(x.float(), mask_ratio=0.75)
        loss, y, mask = model(img)
        process_output(x, y["rgb"], mask["rgb"], model, use_norm, NORM_RGB)
        process_output(x_d, y["depth"], mask["depth"], model, use_norm, NORM_DEPTH)

    def process_output(x, y, mask, model, use_norm, norm):

        y = model.unpatchify(y)
        y = torch.einsum("nchw->nhwc", y).detach().cpu()

        # visualize the mask
        mask = mask.detach()
        print(f"Shape of mask = {mask.shape}")
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
        plt.figure()
        plt.rcParams["figure.figsize"] = [24, 24]

        plt.subplot(1, 4, 1)
        show_image(x[0], use_norm, norm, "original")

        plt.subplot(1, 4, 2)
        show_image(im_masked[0], use_norm, norm, "masked")

        plt.subplot(1, 4, 3)
        show_image(y[0], use_norm, norm, "reconstruction")

        plt.subplot(1, 4, 4)
        show_image(im_paste[0], use_norm, norm, "reconstruction + visible")

    def process_depth_img(depth):
        depth = Image.fromarray((depth / np.max(depth) * 255.0).astype(np.uint8))
        depth = depth.resize((224, 224))
        depth = np.array(depth) / 255.0
        depth = np.stack((depth,) * 3, axis=-1)
        return depth

    # load an image
    data_path = "data/sunrgbd_trainval"
    file_name = str(args.img).zfill(6)
    rgb_path = os.path.join(data_path, "image", file_name + ".jpg")
    depth_path = os.path.join(data_path, "depth", file_name + ".png")
    raw_depth_anything_path = os.path.join(
        data_path, "rawDepthAnything", file_name + ".npy"
    )

    # Preprocess inputs
    rgb = Image.open(rgb_path)
    rgb = rgb.resize((224, 224))
    rgb = np.array(rgb) / 255.0

    depth = np.array(Image.open(depth_path))
    depth = process_depth_img(depth)

    raw_depth_anything = np.load(raw_depth_anything_path)
    raw_depth_anything = process_depth_img(raw_depth_anything)

    # normalize by mean and std
    rgb = rgb - NORM_RGB["mean"]
    rgb = rgb / NORM_RGB["std"]
    depth = depth - NORM_DEPTH["mean"]
    depth = depth / NORM_DEPTH["std"]
    raw_depth_anything = raw_depth_anything - NORM_DEPTH_ANYTHING["mean"]
    raw_depth_anything = raw_depth_anything / NORM_DEPTH_ANYTHING["std"]

    inputs = {"rgb": rgb, "depth": depth, "depth_anything": raw_depth_anything}
    plt.rcParams["figure.figsize"] = [5, 5]

    chkpt_dir = os.path.join(args.ckpt_path, "checkpoint-" + str(args.epoch) + ".pth")
    model_mae = prepare_model(chkpt_dir, "dual_swinmae_s")
    print("Model loaded.")
    torch.manual_seed(2)
    print("MAE with pixel reconstruction:")
    run_model(inputs, model_mae, args.use_norm)
    plt.show()
