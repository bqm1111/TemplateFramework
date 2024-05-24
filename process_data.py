import cv2
import os
import numpy as np
from utils.helper import convert_depth_to_image


raw_depth_path = "data/NYUDepthv2/depth"
raw_depthanything_path = "data/NYUDepthv2/rawDepthAnything"
filename = "12.npy"
raw_depth = np.load(os.path.join(raw_depth_path, filename))
raw_depth_anything = np.load(os.path.join(raw_depthanything_path, filename))
mask = np.where(raw_depth == 0, 0, 1)
# mask = np.stack((mask,) * 3, axis=-1)
# cv2.imshow("before", mask.astype(np.uint8) * 255)
# mask = cv2.blur(mask, (5, 5))
# cv2.imshow("after", mask.astype(np.uint8) * 255)
combine = raw_depth + (1 - mask) * raw_depth_anything

raw_depth = convert_depth_to_image(raw_depth)
raw_depth_anything = convert_depth_to_image(raw_depth_anything)
# inv_mask = cv2.bitwise_not(mask).astype(np.float64)
# mask = mask.astype(np.float64)
# img1 = cv2.multiply(mask, raw_depth.astype(np.float64)) / 255.0
# img2 = cv2.multiply(inv_mask, raw_depth_anything.astype(np.float64)) / 255.0
# result = (cv2.add(img1, img2)).astype(np.uint8)
combine = convert_depth_to_image(combine)

cv2.imshow("depth", raw_depth)
cv2.imshow("depthanything", raw_depth_anything)
cv2.imshow("combine", combine)
cv2.waitKey()
