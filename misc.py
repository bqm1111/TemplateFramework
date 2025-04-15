import torch
import cv2
import numpy as np

image = cv2.imread("data/NYUDepthv2/image/0.jpg")

h, w, c = image.shape
wk = 5
hk = 5
x = np.linspace(0.5, w - 0.5, wk)
y = np.linspace(0.5, h - 0.5, hk)
x, y = np.meshgrid(x, y)
grid = np.stack((x, y), axis=-1)
grid[..., 0] = 2 * grid[..., 0] / (w - 1) - 1
grid[..., 1] = 2 * grid[..., 1] / (h - 1) - 1


def convert_back_to_point(x, w, h):
    return [int((x[0] + 1) * w / 2), int((x[1] + 1) * h / 2)]


grid = grid.reshape(wk * hk, 2)
for g in grid:
    point = convert_back_to_point(g, w, h)
    image = cv2.circle(image, point, 4, color=(0, 0, 255))

cv2.imshow("img", image)
cv2.waitKey()
