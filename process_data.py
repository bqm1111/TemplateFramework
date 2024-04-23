import scipy.io
import cv2
import os
from tqdm import tqdm

path = "/home/sherlock/Pictures/segmentation/Official_SUNRGBD/sunrgbd_trainval/seg_label"
save_path = "/home/sherlock/Pictures/segmentation/Official_SUNRGBD/sunrgbd_trainval/seglabel"
filename = "/home/sherlock/Pictures/segmentation/sunrgbd/sunrgbd_trainval/depth/000005.mat"
image_path = "/home/sherlock/Pictures/segmentation/sunrgbd/sunrgbd_trainval/image/000005.jpg"
img = cv2.imread(image_path)
print(img.shape)

# for filename in tqdm(os.listdir(path)):
order = filename.split(".")[0]
mat = scipy.io.loadmat(os.path.join(path, filename))
depth = mat["instance"]
print(depth.shape)
# print(mat)
    # cv2.imwrite(os.path.join(save_path, order + ".png"), mat["seglabel"])
