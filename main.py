import cv2
import os
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.cluster import DBSCAN
from scipy import optimize


rootdir = "/home/william/extdisk/Data/cybathlon/cppresult"
img_path = "000145_mask.jpg" # 145 or 127
index = img_path[:6]
path_seg  = cv2.imread(os.path.join(rootdir, img_path))
path_seg = cv2.GaussianBlur(path_seg, [3, 3], cv2.BORDER_DEFAULT)
# path_seg = cv2.GaussianBlur(path_seg, [3, 3], cv2.BORDER_DEFAULT)
gx_right_shift = path_seg[:, 1:] - path_seg[:, :-1]
gx_left_shift = path_seg[:, :-1] - path_seg[:, 1:]
gy_down_shift = path_seg[1:, :] - path_seg[:-1, :]
gy_up_shift = path_seg[:-1, :] - path_seg[1:, :]

gx = cv2.bitwise_or(gx_left_shift, gx_right_shift)
gy = cv2.bitwise_or(gy_up_shift, gy_down_shift)

grads = cv2.bitwise_or(gx[1:, :], gy[:, 1:])

kernel_size = 3
kernel = np.ones((kernel_size, kernel_size), np.uint8)
grads_erode = cv2.erode(grads, kernel, iterations=1)

if (len(grads_erode.shape)) == 3:
    H, W, _ = grads_erode.shape
else:
    H, W = grads_erode.shape


thr = 250
valid_points = np.where(grads_erode[:,:,0] > thr)
coordinates = list(zip(valid_points[0], valid_points[1]))

# print("tan(1) = ", np.tan(np.deg2rad(1)))

def l2_norm(a, b):
    return np.linalg.norm(a-b)

def show_pic(img):
    plt.imshow(img)
    plt.show()

def y_coord_transform(h:int, coordinates:list[np.array]):
    """coordinate sample is (y, x), thus we
    change the coordinate system to (h-y, x)
    """
    coord_transform = np.array(coordinates)
    coord_transform[:, 0] = h - coord_transform[:, 0]
    return coord_transform

def y_coord_restore(h:int, transformed_coordinates:np.array):
    """restore from transformed system to original
    """
    coord_restore = np.copy(transformed_coordinates)
    coord_restore[:, 0] = -coord_restore[:, 0] + h
    return coord_restore

def diffusion_method(coordinates:list[np.array], delta_theta=1, begin_angle=0, end_angle=90):
    """this is a classification problem, which aims to classify
    the point is whether at left boundary or right. Therefore we
    construct a vector y=kx (k iterates by delta theta) to find the 
    first intersection and the last along the ray. (this method demands
    a really dense point distribution, in order to avoid missing the point)
    """
    transformed_coords = y_coord_transform(H, coordinates)
    for angle in range(begin_angle, end_angle+1):
        for j in range(W):
            pass


# blank_paper = np.zeros_like(grads_erode, dtype=np.uint8)
# for coord in coordinates:
#     blank_paper[coord[0], coord[1]] = 255

# plt.imshow(blank_paper)
# plt.show()
# tan_theta_valid_info = []
# coord_and_grad = []
# coord = []



# cur_idx = 0
# grads_erode = grads_erode[:, :, 0]
# for i in range(H):
#     cur_x = np.where(grads_erode[i, :] == 255)[0]
#     if cur_x.size > 0:
#         valid_points = (i, cur_x[0])
#         coord.append(valid_points)
#         if len(tan_theta_valid_info) > 0:
#             gy = valid_points[0] - coord[cur_idx - 1][0]
#             gx = valid_points[1] - coord[cur_idx - 1][1]
#             if np.abs(gx) < 1e-6:
#                 tan_theta = np.inf
#             else:
#                 tan_theta = gy / gx
#             tan_theta_valid_info.append(tan_theta)
#             coord_and_grad.append((valid_points, tan_theta))     
#         else:
#             tan_theta_valid_info.append(0)
#             coord_and_grad.append((valid_points, 0))
#         cur_idx += 1

# print(tan_theta_valid_info)
            
        



# fig, axs = plt.subplots(2, 2)
# axs[0, 0].imshow(grads)
# axs[0, 0].axis('off')
# axs[0, 1].imshow(grads_erode)
# axs[0, 1].axis('off')
# axs[1, 0].imshow(gy_down_shift)
# axs[1, 0].axis('off')
# axs[1, 1].imshow(gy_up_shift)
# axs[1, 1].axis('off')
# plt.show()

def apply_dbscan_to_column(column, eps=2, min_samples=2):
    """ Apply DBSCAN clustering to a single column """
    y_coords = np.where(column > 0)[0].reshape(-1, 1)
    if len(y_coords) == 0:
        return None

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(y_coords)
    if len(set(clustering.labels_)) == 1 and -1 in clustering.labels_:
        # No significant cluster found
        return None

    return clustering.labels_, y_coords

def find_middle_lane_dbscan(binary_mask, eps=2, min_samples=2):
    """ Find the middle lane from the binary mask """
    height, width = binary_mask.shape
    middle_lane = np.full(width, -1)  # Initialize with -1 (no lane)

    for i in range(width):
        column = binary_mask[:, i]
        result = apply_dbscan_to_column(column, eps, min_samples)
        if result is not None:
            labels, y_coords = result
            unique_labels = set(labels)
            column_lanes = []

            for label in unique_labels:
                if label != -1:  # Ignore noise
                    lane_points = y_coords[labels == label]
                    lane_center = np.mean(lane_points).astype(int)
                    column_lanes.append(lane_center)

            if column_lanes:
                middle_lane[i] = np.mean(column_lanes).astype(int)

    return middle_lane

def find_middle_lane_rowwise(mask):
    h, w = mask.shape
    middle_lane = np.zeros(h, dtype=np.int32)

    for i in range(h):
        row = mask[i, :]
        white_pixels = np.where(row == 255)[0]
        if white_pixels.size > 0:
            middle_lane[i] = np.mean(white_pixels).astype(np.int32)
    
    return middle_lane

def find_middle_lane_colwise(mask):
    h, w = mask.shape
    middle_lane = np.zeros(w, dtype=np.int32)

    for i in range(w):
        col = mask[:, i]
        white_pixels = np.where(col == 255)[0]
        if white_pixels.size > 0:
            middle_lane[i] = np.mean(white_pixels).astype(np.int32)
    
    return middle_lane


def construct_map(coordinates):
    """change the data-structure from list[np.array] to
    key-value mapping type. For the convinience of searching
    by key
    """
    coordinates_dict = {}
    for coord in coordinates:
        if coord[0] not in coordinates_dict:
            # If not, initialize it with an empty list
            coordinates_dict[coord[0]] = []
        coordinates_dict[coord[0]].append(coord[1])
    return coordinates_dict


def fit_polynomials_scalar(x, coef):
    degrees = len(coef)
    res = 0
    for i in range(degrees):
        res += x**(degrees-i-1)*coef[i]
    return res

def fit_polynomials(coeffs, x):
    return np.polyval(coeffs, x)

def cost_func(domain, target_coeffs, f_coeffs, g_coeffs):
    target = fit_polynomials(target_coeffs, domain)
    f = fit_polynomials(f_coeffs, domain)
    g = fit_polynomials(g_coeffs, domain)

    return np.sum((target - f)**2 + (target - g)**2)

def rotation_method():
    middle_lane = find_middle_lane_rowwise(path_seg[:,:,0])
    middle_lane_coords = []
    for i in range(len(middle_lane)):
        if middle_lane[i] != 0:
            middle_lane_coords.append((i, middle_lane[i]))


    seg_copy = np.copy(path_seg)
    for y, x in middle_lane_coords:
        cv2.circle(seg_copy, (x, y), 1, (0, 255, 0), 2)

    # fit polynomial functions
    middle_lane_coords = np.array(middle_lane_coords)
    func_middle_lane_coef = np.polyfit(middle_lane_coords[:, 0], middle_lane_coords[:, 1], 3)

    for x in range(W):
        y = int(fit_polynomials_scalar(x, func_middle_lane_coef))
        cv2.circle(seg_copy, (y, x), 1, (0, 0, 255), 2)

    path_seg_t = cv2.rotate(path_seg, cv2.ROTATE_90_COUNTERCLOCKWISE)

    middle_lane_t = find_middle_lane_rowwise(path_seg_t[:,:,0])
    middle_lane_coords_t = []
    for i in range(len(middle_lane_t)):
        if middle_lane_t[i] != 0:
            middle_lane_coords_t.append((i, middle_lane_t[i]))

    middle_lane_coords_t = np.array(middle_lane_coords_t)
    middle_lane_coords_t_restore = np.array([[0, -1], [1, 0]]) @ middle_lane_coords_t.T + np.array([[W], [0]])
    # tmp = np.array([[0, -1], [1, 0]])
    # print(tmp.shape)
    # print(middle_lane_coords_t.shape)
    for y, x in middle_lane_coords_t:
        cv2.circle(seg_copy, (np.abs(y-W), x), 1, (255, 0, 0), 2)

    # middle_lane_coords_t = np.array(middle_lane_coords_t)
    # middle_lane_coords_t_restore = np.array(middle_lane_coords_t)
    # middle_lane_coords_t_restore[:, 1] = middle_lane_coords_t[:, 1]
    # middle_lane_coords_t_restore[:, 0] = W - middle_lane_coords_t_restore[:, 0]

    # for coord in middle_lane_coords_t_restore:
    #     cv2.circle(seg_copy, (coord[0], coord[1]), 1, (128, 128, 128), 2)

    middle_lane_coords_t = np.array(middle_lane_coords_t)
    # func_middle_lane_t_coef = np.polyfit(middle_lane_coords_t_restore[:, 1], middle_lane_coords_t_restore[:, 0], 3)
    func_middle_lane_t_coef = np.polyfit(W - middle_lane_coords_t_restore[:, 0], middle_lane_coords_t_restore[:, 1], 3)

    for x in range(W):
        y = int(fit_polynomials_scalar(x, func_middle_lane_t_coef))
        cv2.circle(seg_copy, (y, x), 1, (128, 128, 128), 2)

    # # restore to the same coordinate system
    
    min_1_x = np.min(middle_lane_coords[:, 1])
    max_1_x = np.max(middle_lane_coords[:, 1])
    # # min_1_y = np.min(middle_lane_coords[:, 0])
    # # max_1_y = np.max(middle_lane_coords[:, 0])
    
    # # # vertical_min = np.min(np.abs(middle_lane_coords_t[:, 0]-W))
    # # # vertical_max = np.max(np.abs(middle_lane_coords_t[:, 0]-W))

    min_2_x = np.min(middle_lane_coords_t_restore[:, 0])
    max_2_x = np.max(middle_lane_coords_t_restore[:, 0])
    # print(min_1_x, max_1_x, min_2_x, max_2_x)
    # # min_2_y = np.min(middle_lane_coords[:, 1])
    # # max_2_y = np.max(middle_lane_coords[:, 1])

    domain_x = range(max(min_1_x, min_2_x), min(max_1_x, max_2_x))
    # # domain_y = range(max(min_1_y, min_2_y), min(max_1_y, max_2_y))
    initial_guess = [1, 1, 1, 1]
    
    ideal_curve = optimize.minimize(lambda x: cost_func(domain_x, x, func_middle_lane_coef, func_middle_lane_t_coef), initial_guess)
    print(ideal_curve.x)

    for x in range(W):
        y = int(fit_polynomials_scalar(x, ideal_curve.x))
        cv2.circle(seg_copy, (y, x), 1, (255, 192, 203), 2)

    return seg_copy


if __name__ == "__main__":
    res = rotation_method()
    show_pic(res)
    # cv2.imwrite(f"debug_full_vis.png", res[:,:,::-1])
    import gc
    gc.collect()


