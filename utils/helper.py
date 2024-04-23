import time
import numpy as np
from PIL import Image
import cv2


class Timer():
    def __init__(self) -> None:
        self.start_time = 0.0
        self.end_time = 0.0
        self.start()

    def start(self):
        self.start_time = time.time()

    def end(self, ms=False, clear=False):
        self.end_time = time.time()
        if ms:
            duration = int((self.end_time - self.start_time) * 1000)
        else:
            duration = int(self.end_time - self.start_time)

        if clear:
            self.start_time = time.time()

        return duration


class Average_Meter:
    def __init__(self, keys):
        self.keys = keys
        self.clear()

    def add(self, dic):
        for key, value in dic.items():
            self.data_dic[key].append(value)

    def get(self, keys=None, clear=False):
        if keys is None:
            keys = self.keys

        dataset = {}
        for key in keys:
            dataset[key] = float(np.mean(self.data_dic[key]))

        if clear:
            self.clear()

        return dataset

    def clear(self):
        self.data_dic = {key: [] for key in self.keys}


def show_pil_image(window_name: str, img: Image):
    cv2_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    cv2.imshow(window_name, cv2_image)


if __name__ == '__main__':
    img = Image.open("data/NYUDepthv2/RGB/7.jpg").convert("RGB")
    show_pil_image("img", img)
