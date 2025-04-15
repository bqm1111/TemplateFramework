import os
import cv2
import numpy as np
from tqdm import tqdm
from datasets import get_dataset

import torch
from torch.utils.data import DataLoader
from utils.logger import get_root_logger
from utils.helper import get_class_colors, print_iou
from .metric import hist_info, compute_score

logger = get_root_logger()


class Evaluator:
    def __init__(self, cfg, model, show=False):
        self.show_image = show
        self.cfg = cfg
        self.val_cfg = cfg.val
        self.model = model

        val_dataset = get_dataset(cfg.experiment_dataset, self.val_cfg.dataset)
        self.val_loader = DataLoader(
            val_dataset, batch_size=self.val_cfg.batch_size, shuffle=False
        )
        self.class_names = val_dataset.get_classname()
        self.num_classes = len(self.class_names)
        self.val_logdir = os.path.join(
            self.val_cfg.log_dir,
            self.cfg.experiment_dataset,
            self.cfg.experiment_type,
            self.cfg.experiment_name,
        )
        if "exclude" in self.val_cfg:
            logger.info("ignore some label")
            self.excluded_labels = [
                i
                for i, elem in enumerate(self.class_names)
                if elem in list(self.val_cfg.exclude)
            ]
        else:
            self.excluded_labels = np.array([])

    def run_once(self, model_file, need_load=True):
        if not os.path.exists(self.val_logdir):
            os.makedirs(self.val_logdir)
        with open(os.path.join(self.val_logdir, "result.txt"), "a") as f:
            if not need_load:
                logger.info(f"Evaluate while training")
            else:
                logger.info(f"Loading weight from {model_file}")
                self.model.load_state_dict(
                    torch.load(model_file)["model"], strict=False
                )

        for i, data in enumerate(tqdm(self.val_loader)):
            label = data["label"].squeeze(1)
            pred, shifted_pos, reference_pos = self.eval(data)
            self.visualize(label, data, pred, i, shifted_pos, reference_pos)

    def test_image(self, rgb, depth, idx):
        data = dict()
        data["rgb"] = rgb
        data["depth"] = depth
        pred, shifted_pos, reference_pos = self.eval(data)
        self.visualize(None, data, pred, idx, shifted_pos, reference_pos)

    def convert_back_to_point(self, x, w, h):
        # if int((x[0] + 1) * h / 2) > h:
        #     print(x)
        #     print(int((x[0] + 1) * h / 2))
        #     print("out of frame")
        return [int((x[1] + 1) * w / 2), int((x[0] + 1) * h / 2)]

    def run(self, model_file, need_load=True):
        if not os.path.exists(self.val_logdir):
            os.makedirs(self.val_logdir)
        with open(os.path.join(self.val_logdir, "result.txt"), "a") as f:
            if not need_load:
                logger.info(f"Evaluate while training")
            else:
                logger.info(f"Loading weight from {model_file}")
                self.model.load_state_dict(
                    torch.load(model_file)["model"], strict=False
                )
            all_results = []

            for idx, data in enumerate(tqdm(self.val_loader)):
                label = data["label"].squeeze(1)
                pred, shifted_pos, reference_pos = self.eval(data)
                pred_trunc = np.array(pred.cpu())
                pred_trunc[pred_trunc >= self.num_classes] = self.num_classes
                hist_tmp, labeled_tmp, correct_tmp = hist_info(
                    self.num_classes,
                    pred_trunc,
                    np.array(label.cpu()),
                    excluded_labels=self.excluded_labels,
                )
                results_dict = {
                    "hist": hist_tmp,
                    "labeled": labeled_tmp,
                    "correct": correct_tmp,
                }
                self.visualize(label, data, pred, idx, shifted_pos, reference_pos)
                all_results.append(results_dict)

            result_line, meanIoU = self.compute_metric(all_results)
            f.write("Model: " + str(model_file) + "\n")
            f.write(result_line)
            f.write("\n")
            f.flush()
        return meanIoU

    def compute_metric(self, results):
        hist = np.zeros((self.num_classes, self.num_classes))
        correct = 0
        labeled = 0
        count = 0
        for d in results:
            hist += d["hist"]
            correct += d["correct"]
            labeled += d["labeled"]
            count += 1

        iou, mean_IoU, _, freq_IoU, mean_pixel_acc, pixel_acc = compute_score(
            hist, correct, labeled
        )
        result_line = print_iou(
            iou,
            freq_IoU,
            mean_pixel_acc,
            pixel_acc,
            self.class_names,
            show_no_back=False,
        )
        return result_line, mean_IoU

    def visualize(
        self, label, data, pred, idx=None, shifted_pos=None, reference_pos=None
    ):
        NORM_RGB = {
            "mean": np.array([0.485, 0.456, 0.406]),
            "std": np.array([0.229, 0.224, 0.225]),
        }
        # Get color corresponding to each classes
        colors = np.array(get_class_colors(self.num_classes + 1))

        # Convert data to unit8 numpy type on cpu
        pred_arr = pred.squeeze(0).cpu().numpy().astype(np.uint8)
        rgb_arr = data["rgb"].squeeze(0).cpu().numpy()
        pred_arr[pred_arr > self.num_classes] = self.num_classes
        if data["depth"] is not None:
            depth_arr = data["depth"].squeeze(0).cpu().numpy()
        colored_pred = np.zeros_like(pred_arr)
        colored_pred = np.stack((colored_pred,) * 3, axis=-1)
        colored_pred[:] = colors[pred_arr[:]]

        if label is not None:
            label_arr = label.squeeze(0).cpu().numpy().astype(np.uint8)
            # Convert void classes label from 255 to self.num_classes
            label_arr[label_arr == 255] = self.num_classes
            # Colorize groundtruth and prediction
            colored_label = np.zeros_like(label_arr)
            colored_label = np.stack((colored_label,) * 3, axis=-1)
            colored_label[:] = colors[label_arr[:]]
        else:
            colored_label = colored_pred

        # Overlay prediction on input images
        rgb_arr = rgb_arr.transpose(1, 2, 0)
        rgb_arr = ((rgb_arr * NORM_RGB["std"] + NORM_RGB["mean"]) * 255).astype(
            np.uint8
        )
        rgb_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
        depth_arr = (depth_arr.transpose(1, 2, 0) * 255).astype(np.uint8)
        depth_arr = cv2.cvtColor(depth_arr, cv2.COLOR_RGB2BGR)
        offset_vis = depth_arr.copy()
        # print("Offset")
        # for p in shifted_pos:
        #     print(p.shape)
        # print("Reference")
        # for p in reference_pos:
        #     print(p.shape)

        deformed_grid = np.array(shifted_pos[1][1].cpu())
        print("grid shape = ", deformed_grid.shape)
        deformed_grid = deformed_grid.reshape(300, 2)
        reference_grid = np.array(reference_pos[1][1].cpu())
        reference_grid = reference_grid.reshape(300, 2)
        refs = []
        deforms = []
        for g in deformed_grid:
            point = self.convert_back_to_point(g, 640, 480)
            deforms.append(point)
            offset_vis = cv2.circle(offset_vis, point, 4, color=(0, 0, 255))
        for g in reference_grid:
            point = self.convert_back_to_point(g, 640, 480)
            refs.append(point)
            offset_vis = cv2.circle(offset_vis, point, 4, color=(0, 255, 0))

        for ref, defo in zip(np.array(refs), np.array(deforms)):
            cv2.arrowedLine(
                offset_vis, ref, defo, color=(255, 0, 0), thickness=2, tipLength=0.1
            )

        # Concatenate multiple outputs for saving
        # output = np.concatenate(
        #     [rgb_arr, depth_arr, colored_label, colored_pred], axis=1
        # )
        output = np.concatenate([rgb_arr, offset_vis], axis=1)

        # Save results
        if self.val_cfg.save_path is not None:
            if not os.path.exists(self.val_cfg.save_path):
                os.makedirs(self.val_cfg.save_path)
            cv2.imwrite(
                os.path.join(self.val_cfg.save_path, str(idx).zfill(6) + ".png"), output
            )

        # Show results
        if self.show_image:
            cv2.imshow("pred", output)
            if cv2.waitKey() == ord("q"):
                exit(0)

    def eval(self, data):
        self.model.eval()
        for key, value in data.items():
            data[key] = value.cuda()
        self.model = self.model.cuda()
        with torch.no_grad():
            score, pos, ref = self.model(data["rgb"], data["depth"])
        pred = score.argmax(1)
        return pred, pos, ref
