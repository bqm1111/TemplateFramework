import time
import os
import cv2
import numpy as np
from tqdm import tqdm
from datasets import get_dataset

import torch
from torch.utils.data import DataLoader
from utils.logger import get_root_logger
from PIL import Image
from visualization.visualize import get_class_colors, show_img, print_iou
from datasets import get_classname
from .metric import hist_info, compute_score
logger = get_root_logger()


class Evaluator:
    def __init__(self, cfg, model, show=False):
        self.show_image = show
        self.cfg = cfg
        self.val_cfg = cfg.val
        self.model = model

        val_dataset = get_dataset(self.val_cfg.dataset)
        self.val_loader = DataLoader(
            val_dataset, batch_size=self.val_cfg.batch_size, shuffle=False)
        self.class_names = get_classname(self.val_cfg.dataset.name)
        self.num_classes = len(self.class_names)
        self.val_logdir = os.path.join(
            self.val_cfg.log_dir, self.cfg.experiment_type, self.cfg.experiment_dataset, self.cfg.experiment_name)

    def run_once(self, model_file, img_index=0,  need_load=True):
        if not os.path.exists(self.val_logdir):
            os.makedirs(self.val_logdir)
        with open(os.path.join(self.val_logdir, "result.txt"), 'a') as f:
            if not need_load:
                logger.info(f"Evaluate while training")
            else:
                logger.info(f"Loading weight from {model_file}")
                self.model.load_state_dict(torch.load(model_file)["model"])
        for i, data in enumerate(self.val_loader):
            if i == img_index:
                label = data["label"].squeeze(1)
                pred, pos = self.eval(data)
                print(f"pos now shape = {pos[0][0].shape}")
                self.visualize(label, data, pred, pos)
                break

    def run(self, model_file, need_load=True):
        if not os.path.exists(self.val_logdir):
            os.makedirs(self.val_logdir)
        with open(os.path.join(self.val_logdir, "result.txt"), 'a') as f:
            if not need_load:
                logger.info(f"Evaluate while training")
            else:
                logger.info(f"Loading weight from {model_file}")
                self.model.load_state_dict(torch.load(model_file)["model"])
            all_results = []
            for _, data in enumerate(tqdm(self.val_loader)):
                label = data["label"].squeeze(1)
                start = time.time()
                pred = self.eval(data)
                end = time.time()
                logger.info(f"Eval time = {(end - start) * 1000}")
                hist_tmp, labeled_tmp, correct_tmp = hist_info(
                    self.num_classes, np.array(pred.cpu()), np.array(label.cpu()))
                results_dict = {
                    "hist": hist_tmp,
                    "labeled": labeled_tmp,
                    "correct": correct_tmp,
                }
                self.visualize(label, data, pred)
                all_results.append(results_dict)

            result_line, meanIoU = self.compute_metric(all_results)
            f.write('Model: ' + str(model_file) + '\n')
            f.write(result_line)
            f.write('\n')
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
            hist, correct, labeled)
        result_line = print_iou(
            iou,
            freq_IoU,
            mean_pixel_acc,
            pixel_acc,
            self.class_names,
            show_no_back=False,
        )
        return result_line, mean_IoU

    def convert_back_to_point(self, x, w, h):
        return [int((x[1] + 1) * w/2), int((x[0] + 1) * h / 2)]

    def visualize(self, label, data, pred, pos):
        if self.val_cfg.save_path is not None:
            if not os.path.exists(self.val_cfg.save_path):
                os.makedirs(self.val_cfg.save_path)
                os.makedirs(self.val_cfg.save_path + "_color")

            fn = "test.png"

            # save colored result
            result_img = Image.fromarray(pred.astype(np.uint8), mode="P")
            class_colors = get_class_colors(self.num_classes)
            palette_list = list(np.array(class_colors).flat)
            if len(palette_list) < 768:
                palette_list += [0] * (768 - len(palette_list))
            result_img.putpalette(palette_list)
            result_img.save(os.path.join(self.save_path + "_color", fn))

            # save raw result
            cv2.imwrite(os.path.join(self.save_path, fn), pred)
            logger.info("Save the image " + fn)

        if self.show_image:
            NORM_RGB = {
                "mean": np.array([0.485, 0.456, 0.406]),
                "std": np.array([0.229, 0.224, 0.225]),
            }
            colors = np.array(get_class_colors(self.num_classes))
            pred_arr = pred.squeeze(0).cpu().numpy().astype(np.uint8)
            rgb_arr = data["rgb"].squeeze(0).cpu().numpy()
            depth_arr = data["depth"].squeeze(0).cpu().numpy() * 255
            depth_arr = depth_arr.transpose(1, 2, 0).astype(np.uint8)
            depth_arr = cv2.cvtColor(depth_arr, cv2.COLOR_RGB2BGR)
            for p in pos:
                print(p.shape)
            grid = np.array(pos[3][4].cpu())
            grid = grid.reshape(300, 2)
            for g in grid:
                point = self.convert_back_to_point(g, 640, 480)
                depth_arr = cv2.circle(depth_arr, point, 4, color=(0, 0, 255))

            colored_pred = np.zeros_like(pred_arr)
            colored_pred = np.stack((colored_pred,)*3, axis=-1)
            colored_pred[:] = colors[pred_arr[:]]
            rgb_arr = rgb_arr.transpose(1, 2, 0)
            rgb_arr = (
                (rgb_arr * NORM_RGB["std"] + NORM_RGB["mean"]) * 255).astype(np.uint8)
            rgb_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
            dst = cv2.addWeighted(rgb_arr, 0.5, colored_pred, 0.5, 0.0)
            output = np.concatenate([dst, rgb_arr, colored_pred], axis=1)

            # cv2.imshow("pred", output)
            cv2.imshow("img", depth_arr)

            if cv2.waitKey() == ord('q'):
                exit(0)

    def eval(self, data):
        self.model.eval()
        for key, value in data.items():
            data[key] = value.cuda()
        self.model = self.model.cuda()
        with torch.no_grad():
            score, pos = self.model(data["rgb"], data["depth"])
        pred = score.argmax(1)
        return pred, pos
