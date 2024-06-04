import cv2
from visualization.visualize import get_class_colors
from datasets.class_names import nyuv2_classname
import numpy as np

class_name = nyuv2_classname()
colors = get_class_colors(len(class_name))
img = np.zeros((480, 1280, 3))
img[:] = [128, 128, 160]
h, w, c = img.shape
distance_x = w // 9
num_class = len(class_name)
start_x = (w - distance_x * 7) // 2
start_y = 100
distance_y = 80
box_w = 10
box_h = 10
for i in range(num_class):
    y = i // 8
    x = i % 8
    center_x = start_x + x * distance_x
    center_y = start_y + y * distance_y
    text_length = len(class_name[i])
    cv2.putText(img, class_name[i], (center_x - text_length // 2 * 8, center_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA, False)
    cv2.rectangle(img, (center_x - box_w, center_y - box_h),
                  (center_x + box_w, center_y + box_h), colors[i], -1)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = img.astype(np.uint8)
cv2.imshow("img", img)
cv2.waitKey()
