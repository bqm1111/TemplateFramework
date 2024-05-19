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
logger = get_root_logger()


class Evaluator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.val_cfg = cfg.val
        self.model = get_model(cfg.model.name, **cfg.model.params)

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
            # self.model.load_state_dict(torch.load(model_file)["model"])
            result_line = self.single_process_evaluation()
            f.write('Model: ' + self.cfg.model.name + '\n')
            f.write(result_line)
            f.write('\n')
            f.flush()

    def single_process_evaluation(self):
        all_results = []
        for idx, data in enumerate(tqdm(self.val_loader)):
            results_dict = self.func_per_iteration(idx, data)
            all_results.append(results_dict)

        result_line = self.compute_metric(all_results)
        return result_line

    @staticmethod
    def hist_info(n_cl, pred, gt):
        assert (pred.shape == gt.shape)
        k = (gt >= 0) & (gt < n_cl)
        labeled = np.sum(k)
        correct = np.sum((pred[k] == gt[k]))
        confusionMatrix = np.bincount(n_cl * gt[k].astype(int) + pred[k].astype(int),
                                      minlength=n_cl ** 2).reshape(n_cl, n_cl)
        return confusionMatrix, labeled, correct

    @staticmethod
    def compute_score(hist, correct, labeled):
        iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
        mean_IoU = np.nanmean(iou)
        mean_IoU_no_back = np.nanmean(iou[1:])  # useless for NYUDv2

        freq = hist.sum(1) / hist.sum()
        freq_IoU = (iou[freq > 0] * freq[freq > 0]).sum()

        classAcc = np.diag(hist) / hist.sum(axis=1)
        mean_pixel_acc = np.nanmean(classAcc)

        pixel_acc = correct / labeled

        return iou, mean_IoU, mean_IoU_no_back, freq_IoU, mean_pixel_acc, pixel_acc

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

        iou, mean_IoU, _, freq_IoU, mean_pixel_acc, pixel_acc = self.compute_score(
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
        return result_line

    def func_per_iteration(self, iter, data):
        label = data["label"]
        # pred = self.eval(data)
        pred = label
        hist_tmp, labeled_tmp, correct_tmp = self.hist_info(
            self.num_classes, np.array(pred), np.array(label))
        results_dict = {
            "hist": hist_tmp,
            "labeled": labeled_tmp,
            "correct": correct_tmp,
        }
        if self.val_cfg.save_path is not None:
            if not os.path.exists(self.cfg.save_path):
                os.makedirs(self.cfg.save_path)
                os.makedirs(self.cfg.save_path + "_color")

            fn = str(iter) + ".png"

            # save colored result
            result_img = Image.fromarray(pred.astype(np.uint8), mode="P")
            class_colors = get_class_colors(self.num_classes + 1)
            palette_list = list(np.array(class_colors).flat)
            if len(palette_list) < 768:
                palette_list += [0] * (768 - len(palette_list))
            result_img.putpalette(palette_list)
            result_img.save(os.path.join(self.save_path + "_color", fn))

            # save raw result
            cv2.imwrite(os.path.join(self.save_path, fn), pred)
            logger.info("Save the image " + fn)

        if self.val_cfg.show_image:
            colors = get_class_colors(self.num_classes + 1)
            image = data["rgb"].squeeze(0).cpu().numpy()
            image = image.transpose(1, 2, 0)
            print(pred.type)
            cv2.imshow("img", image)
            cv2.imshow("pred", pred.squeeze(0).numpy())
            cv2.waitKey()
            clean = np.zeros(label.shape)
            comp_img = show_img(colors, self.val_cfg.background,
                                image, clean, label, pred)
            cv2.imshow("comp_image", comp_img)
            cv2.waitKey(0)

        return results_dict

    def eval(self, data):
        self.model.eval()
        self.model.to(data["rgb"].get_device())
        with torch.no_grad():
            score = self.model(data)
            pred = score[0]
        pred = pred.argmax(2)
        return pred
    