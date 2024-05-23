import cv2
import os
import numpy as np
from utils.helper import convert_depth_to_image


raw_depth_path = "data/NYUDepthv2/depth"
raw_depthanything_path = "data/NYUDepthv2/rawDepthAnything"
filename = "31.npy"
raw_depth = np.load(os.path.join(raw_depth_path, filename))
raw_depth_anything = np.load(os.path.join(raw_depthanything_path, filename))
mask = np.where(raw_depth == 0, 0, 1)
combine = raw_depth + (1 - mask) * raw_depth_anything

raw_depth = convert_depth_to_image(raw_depth)
raw_depth_anything = convert_depth_to_image(raw_depth_anything)
combine = convert_depth_to_image(combine)

cv2.imshow("depth", raw_depth)
cv2.imshow("depthanything", raw_depth_anything)
cv2.imshow("combine", combine)
cv2.waitKey()
