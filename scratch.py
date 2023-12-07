import numpy as np
import cv2
import os

# rootdir = "/home/william/extdisk/Data/cybathlon/cppresult"
# img_path = "000145_mask.jpg" #or 127
# path_seg  = cv2.imread(os.path.join(rootdir, img_path))
# path_seg = cv2.GaussianBlur(path_seg, [3, 3], cv2.BORDER_DEFAULT)
# # path_seg = cv2.GaussianBlur(path_seg, [3, 3], cv2.BORDER_DEFAULT)
# gx_right_shift = path_seg[:, 1:] - path_seg[:, :-1]
# gx_left_shift = path_seg[:, :-1] - path_seg[:, 1:]
# gy_down_shift = path_seg[1:, :] - path_seg[:-1, :]
# gy_up_shift = path_seg[:-1, :] - path_seg[1:, :]

# gx = cv2.bitwise_or(gx_left_shift, gx_right_shift)
# gy = cv2.bitwise_or(gy_up_shift, gy_down_shift)

# grads = cv2.bitwise_or(gx[1:, :], gy[:, 1:])

# kernel_size = 3
# kernel = np.ones((kernel_size, kernel_size), np.uint8)
# grads_erode = cv2.erode(grads, kernel, iterations=1)

# if (len(grads_erode.shape)) == 3:
#     H, W, _ = grads_erode.shape
# else:
#     H, W = grads_erode.shape


# thr = 250
# valid_points = np.where(grads_erode[:,:,0] > thr)
# coordinates = list(zip(valid_points[0], valid_points[1]))

# # make sure coordinates is sorted by coordinates[0]
# coordinates_dict = {}
# for coord in coordinates:
#     if coord[0] not in coordinates_dict:
#         # If not, initialize it with an empty list
#         coordinates_dict[coord[0]] = []
#     coordinates_dict[coord[0]].append(coord[1])

# print(coordinates_dict)

string = "000127"
print(int(string))