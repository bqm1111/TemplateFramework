import warnings

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
from torchvision.models.segmentation import deeplabv3_resnet50
import torch
import torch.functional as F
import numpy as np
import requests
import torchvision
from PIL import Image
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam import GradCAM
import cv2

import os
import argparse

from omegaconf import OmegaConf
from engine.evaluator import Evaluator
from engine import get_model
import torch
from utils.helper import convert_depth_to_three_channel_img, get_class_colors
import numpy as np
import copy
import cv2


def visualize(data, pred, num_classes=40):
    NORM_RGB = {
        "mean": np.array([0.485, 0.456, 0.406]),
        "std": np.array([0.229, 0.224, 0.225]),
    }
    # Get color corresponding to each classes
    colors = np.array(get_class_colors(num_classes + 1))

    # Convert data to unit8 numpy type on cpu
    pred_arr = pred.squeeze(0).cpu().numpy().astype(np.uint8)
    rgb_arr = data["rgb"].squeeze(0).cpu().numpy()
    pred_arr[pred_arr > num_classes] = num_classes
    if data["depth"] is not None:
        depth_arr = data["depth"].squeeze(0).cpu().numpy()
    colored_pred = np.zeros_like(pred_arr)
    colored_pred = np.stack((colored_pred,) * 3, axis=-1)
    colored_pred[:] = colors[pred_arr[:]]

    # Overlay prediction on input images
    rgb_arr = rgb_arr.transpose(1, 2, 0)
    rgb_arr = ((rgb_arr * NORM_RGB["std"] + NORM_RGB["mean"]) * 255).astype(np.uint8)
    rgb_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
    depth_arr = (depth_arr.transpose(1, 2, 0) * 255).astype(np.uint8)

    # Concatenate multiple outputs for saving
    output = np.concatenate([rgb_arr, depth_arr, colored_pred], axis=1)

    cv2.imshow("pred", output)
    if cv2.waitKey() == ord("q"):
        exit(0)


def setup_model(cfg_file, device):
    config = OmegaConf.load(cfg_file)
    model = get_model(config.model.name, eval=True, **config.model.params)
    checkpoint_path = os.path.join(
        "output_dir/",
        config.experiment_dataset,
        config.experiment_name,
        "checkpoint-" + str(500) + ".pth",
    )

    model.load_state_dict(torch.load(checkpoint_path)["model"], strict=False)
    model = model.to(device)
    return model


def preprocess(input_rgb, input_depth):
    depth = copy.copy(input_depth)
    rgb = copy.copy(input_rgb)
    depth[np.isnan(depth)] = 0  # Replace NaN with 0
    depth[np.isinf(depth)] = 0  # Replace Inf with 0

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    rgb = (rgb / 255.0 - mean) / std

    depth = convert_depth_to_three_channel_img(depth) / 255.0
    rgb = rgb.transpose(2, 0, 1)
    depth = depth.transpose(2, 0, 1)

    rgb = torch.from_numpy(rgb).unsqueeze(0).cuda().float()
    depth = torch.from_numpy(depth).unsqueeze(0).cuda().float()
    rgb.requires_grad_(True)
    depth.requires_grad_(True)

    output = {"rgb": rgb, "depth": depth}

    return output


def predict(model, rgb, depth):
    data = preprocess(rgb, depth)
    with torch.no_grad():
        score = model.sampling(data["rgb"], data["depth"])
    pred = score.argmax(1)

    return pred


class SegmentationModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(SegmentationModelWrapper, self).__init__()
        self.model = model
        self.rgb_input = None
        self.depth_input = None

    def forward(self, x):
        # `x` is a dummy input (not used directly); we use stored inputs instead
        assert (
            self.rgb_input is not None and self.depth_input is not None
        ), "Inputs not set!"
        assert (
            self.rgb_input is not None and self.depth_input is not None
        ), "Inputs not set!"
        # Ensure inputs are still attached to the graph
        assert (
            self.rgb_input.requires_grad and self.depth_input.requires_grad
        ), "Inputs must require grad!"
        score, _, _ = self.model(self.rgb_input, self.depth_input)
        pred = score.argmax(1)

        return score

    def set_inputs(self, rgb_input, depth_input):
        self.rgb_input = rgb_input
        self.depth_input = depth_input


# Custom target for segmentation
class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()

    def __call__(self, model_output):
        return (model_output[self.category, :, :] * self.mask).sum()


if __name__ == "__main__":
    from PIL import Image
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=int, help="img index")
    args = parser.parse_args()

    cfg_file = "config/semseg/nyuv2/dual_dat_small_uper.yaml"
    rgb = Image.open("data/NYUDepthv2/image/" + str(args.img) + ".jpg").convert("RGB")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    rgb = np.array(rgb)
    depth = np.load("data/NYUDepthv2/depth/" + str(args.img) + ".npy")
    data = preprocess(rgb, depth)

    model = setup_model(cfg_file, device)
    # for layer in model.modules():
    #     print(layer)
    # print(model.backbone.stages_d[3].attns)
    print(model.backbone.FFMs[3])
    model.eval()
    model = SegmentationModelWrapper(model)
    # target_layer = model.model.encoder.last_conv
    model.set_inputs(data["rgb"], data["depth"])
    dummy_input = torch.zeros(
        1, 3, 256, 256, requires_grad=True
    )  # Shape doesn’t matter much here
    output = model(dummy_input)
    normalized_masks = torch.nn.functional.softmax(output, dim=1).cpu()
    sem_classes = [
        "wall",
        "floor",
        "cabinet",
        "bed",
        "chair",
        "sofa",
        "table",
        "door",
        "window",
        "bookshelf",
        "picture",
        "counter",
        "blinds",
        "desk",
        "shelves",
        "curtain",
        "dresser",
        "pillow",
        "mirror",
        "floor mat",
        "clothes",
        "ceiling",
        "books",
        "refridgerator",
        "television",
        "paper",
        "towel",
        "shower curtain",
        "box",
        "whiteboard",
        "person",
        "night stand",
        "toilet",
        "sink",
        "lamp",
        "bathtub",
        "bag",
        "otherstructure",
        "otherfurniture",
        "otherprop",
    ]

    sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}

    semantic_category = sem_class_to_idx["bookshelf"]
    car_mask = normalized_masks[0, :, :, :].argmax(axis=0).detach().cpu().numpy()
    car_mask_uint8 = 255 * np.uint8(car_mask == semantic_category)
    semantic_mask_float = np.float32(car_mask == semantic_category)

    both_images = np.hstack(
        (
            rgb,
            np.repeat(car_mask_uint8[:, :, None], 3, axis=-1),
        )
    )
    both_images = cv2.cvtColor(both_images, cv2.COLOR_RGB2BGR)
    cv2.imshow("depth", depth)
    cv2.imshow("mask", both_images)
    # target_layers = [model.model.backbone.stages[3].attns[0].proj_out]
    target_layers = [model.model.backbone.FFMs[2]]

    targets = [SemanticSegmentationTarget(semantic_category, semantic_mask_float)]
    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=dummy_input, targets=targets)[0, :]
        grayscale_cam = cv2.resize(
            grayscale_cam, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR
        )
        cam_image = show_cam_on_image(rgb / 255.0, grayscale_cam, use_rgb=True)

    img = Image.fromarray(cam_image)
    cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
    cv2.imshow("cam", cam_image)
    cv2.waitKey()
