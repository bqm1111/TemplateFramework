import enum
import os
import cv2
from matplotlib import axis
import numpy as np
from tqdm import tqdm
from datasets import get_dataset
from engine import get_model

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
        self.val_logdir = self.val_cfg.log_dir

    def run(self, model_file):
        if not os.path.exists(self.val_logdir):
            os.makedirs(self.val_logdir)
        with open(os.path.join(self.val_logdir, "result.txt"), 'a') as f:
            logger.info(f"Loading weight from {model_file}")
            self.model.load_state_dict(torch.load(model_file)["model"])
            all_results = []
            for _, data in enumerate(tqdm(self.val_loader)):
                label = data["label"]
                pred = self.eval(data)
                hist_tmp, labeled_tmp, correct_tmp = hist_info(
                    self.num_classes, np.array(pred.cpu()), np.array(label.cpu()))
                results_dict = {
                    "hist": hist_tmp,
                    "labeled": labeled_tmp,
                    "correct": correct_tmp,
                }
                self.visualize(label, data, pred)
                all_results.append(results_dict)

            result_line = self.compute_metric(all_results)
            f.write('Model: ' + self.cfg.model.name + '\n')
            f.write(result_line)
            f.write('\n')
            f.flush()

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
        return result_line

    def visualize(self, label, data, pred):
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
            colors = np.array(get_class_colors(self.num_classes))
            pred_arr = pred.squeeze(0).cpu().numpy().astype(np.uint8)
            img = np.zeros_like(pred_arr)
            img = np.stack((img,)*3, axis=-1)
            img[:] = colors[pred_arr[:]]
            cv2.imshow("pred", img)

            if cv2.waitKey() == ord('q'):
                exit(0)

    def eval(self, data):
        self.model.eval()
        for key, value in data.items():
            data[key] = value.cuda()
        self.model = self.model.cuda()
        with torch.no_grad():
            score = self.model(data["rgb"], data["depth"])
        pred = score.argmax(1)
        return pred
