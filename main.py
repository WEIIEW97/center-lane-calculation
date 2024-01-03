import cv2
import os
import sys
# if sys.platform == "linux":
#     os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
from skimage.morphology import skeletonize, medial_axis
from skimage.measure import find_contours
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.cluster import DBSCAN
from scipy import optimize
from scipy.interpolate import interp1d, CubicSpline
from scipy.signal import convolve2d
from sklearn import svm
import glob
import re
from typing import List, Tuple

def gaussian_blur(img, kernel_size=[3, 3]):
    return cv2.GaussianBlur(img, kernel_size, cv2.BORDER_DEFAULT)

def calculate_kernel_radius(binary_mask, thr=255):
    # m = np.where(binary_mask == thr)
    m = binary_mask == thr
    m = m.astype(np.uint8)
    rowsum = np.sum(m, axis=1)
    filter_out_idx = np.where(rowsum > 0)
    chosen_row = rowsum[filter_out_idx]
    mean_val = np.mean(chosen_row)
    radius = int(mean_val / 4)
    return radius - 1 if radius % 2 == 0 else radius


def extract_boundary_line(binary_mask, thr=250):
    gx_right_shift = binary_mask[:, 1:] - binary_mask[:, :-1]
    gx_left_shift = binary_mask[:, :-1] - binary_mask[:, 1:]
    gy_down_shift = binary_mask[1:, :] - binary_mask[:-1, :]
    gy_up_shift = binary_mask[:-1, :] - binary_mask[1:, :]

    # make gx and gy wider
    gx = cv2.bitwise_or(gx_left_shift, gx_right_shift)
    gy = cv2.bitwise_or(gy_up_shift, gy_down_shift)

    grads = cv2.bitwise_or(gx[1:, :], gy[:, 1:])

    kernel_size = 3
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # only erodes once to avoid too much erosion
    grads_erode = cv2.erode(grads, kernel, iterations=1)
    valid_points = np.where(grads_erode[:,:,0] > thr)
    coordinates = list(zip(valid_points[0], valid_points[1]))
    return coordinates

def get_filenames(rootdir:str, specifier:str):
    conditional_filenames = glob.glob(rootdir + f"*{specifier}*")
    return conditional_filenames

def make_skeleton(binary_image, level=0.8):
    skeleton = skeletonize(binary_image, method="lee")
    c = find_contours(skeleton, level=level)
    return c

def calculate_medial_axis(binary_image, return_distance=False):
    results = medial_axis(binary_image, return_distance=return_distance)
    return results

def l2_norm(a, b):
    return np.linalg.norm(a-b)

def show_pic(img):
    plt.imshow(img)
    plt.show()

def y_coord_transform(h:int, coordinates:List[np.array]):
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

def diffusion_method(coordinates:List[np.array], h, w, delta_theta=1, begin_angle=0, end_angle=90):
    """this is a classification problem, which aims to classify
    the point is whether at left boundary or right. Therefore we
    construct a vector y=kx (k iterates by delta theta) to find the 
    first intersection and the last along the ray. (this method demands
    a really dense point distribution, in order to avoid missing the point)
    """
    transformed_coords = y_coord_transform(h, coordinates)
    for angle in range(begin_angle, end_angle+1):
        for j in range(w):
            pass


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

def scanline_threshold_selection(coordinates:List[np.array], tolerance=10):
    coord_map = construct_map(coordinates)
    x_distance = []
    for k in coord_map.keys():
        pixels = coord_map[k]
        if len(pixels) > 1:
            min_val = np.min(pixels)
            max_val = np.max(pixels)
            x_diff = max_val - min_val
            x_distance.append((k, x_diff))

    x_distance_array = np.array(x_distance)
    # print(x_distance_array)

    diff_median = np.median(x_distance_array[:, 1])
    diff_mean = np.mean(x_distance_array[:, 1])
    diff_max = np.max(x_distance_array[:, 1])
    diff_min = np.min(x_distance_array[:, 1])

    if np.abs(diff_median - diff_mean) > tolerance:
        if np.abs(diff_median - diff_min) < np.abs(diff_median - diff_max):
            selected = np.where(x_distance_array[:, 1] < diff_mean)
        else:
            selected = np.where(x_distance_array[:, 1] > diff_mean)
        x_distance_array = x_distance_array[selected]
    return x_distance_array

def find_middle_lane_rowwise(mask):
    if len(mask.shape) == 3:
        h, w, _ = mask.shape
    else:
        h, w = mask.shape
    middle_lane = np.zeros(h, dtype=np.int32)

    for i in range(h):
        row = mask[i, :]
        white_pixels = np.where(row == 255)[0]
        if white_pixels.size > 0:
            middle_lane[i] = np.mean(white_pixels).astype(np.int32)
    
    middle_lane_coords = []
    for i in range(len(middle_lane)):
        if middle_lane[i] != 0:
            middle_lane_coords.append((middle_lane[i], i))

    return middle_lane_coords


def calculate_middle_lane(mask, coordinates, tolerance=10):
    selection = scanline_threshold_selection(coordinates, tolerance)
    rough_middle_lane = find_middle_lane_rowwise(mask)
    rough_middle_lane = np.array(rough_middle_lane)
    optimized_middle_lane = []
    for i in range(len(selection)):
        optimized_middle_lane.append((selection[i][0], rough_middle_lane[selection[i][0]]))
    return optimized_middle_lane


def find_middle_lane_colwise(mask):
    if len(mask.shape) == 3:
        h, w, _ = mask.shape
    else:
        h, w = mask.shape
    middle_lane = np.zeros(w, dtype=np.int32)

    for i in range(w):
        col = mask[:, i]
        white_pixels = np.where(col == 255)[0]
        if white_pixels.size > 0:
            middle_lane[i] = np.mean(white_pixels).astype(np.int32)

    middle_lane_coords = []
    for j in range(len(middle_lane)):
        if middle_lane[j] != 0:
            middle_lane_coords.append((middle_lane[i], j))
    
    return middle_lane_coords


def radius_check(coordinates, central_point):
    map_coord = construct_map(coordinates)
    left_sparse_boundary_points = []
    right_sparse_boundary_points = []
    for key in map_coord.keys():
        horizontal_line_points = map_coord[key]
        min_x_point = np.min(horizontal_line_points)
        max_x_point = np.max(horizontal_line_points)

        r1 = l2_norm(np.array(central_point), np.array([key, min_x_point]))
        r2 = l2_norm(np.array(central_point), np.array([key, max_x_point]))
        
        if r1 < r2:
            left_sparse_boundary_points.append((key, min_x_point))
            right_sparse_boundary_points.append((key, max_x_point))
        else:
            left_sparse_boundary_points.append((key, max_x_point))
            right_sparse_boundary_points.append((key, min_x_point))
    return left_sparse_boundary_points, right_sparse_boundary_points

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

def rotation_method(path_seg:np.array):
    if len(path_seg.shape) == 3:
        h, w, _ = path_seg.shape
    else:
        h, w = path_seg.shape

    middle_lane = find_middle_lane_rowwise(path_seg)
    middle_lane_coords = []
    for i in range(len(middle_lane)):
        if middle_lane[i] != 0:
            middle_lane_coords.append((i, middle_lane[i]))

    seg_copy = np.copy(path_seg)
    for y, x in middle_lane_coords:
        cv2.circle(seg_copy, (x, y), 1, (0, 255, 0), 2)

    # fit polynomial functions
    middle_lane_coords = np.array(middle_lane_coords)
    func_middle_lane_coef = np.polyfit(middle_lane_coords[:, 1], middle_lane_coords[:, 0], 5)

    for x in range(w):
        y = int(fit_polynomials_scalar(x, func_middle_lane_coef))
        cv2.circle(seg_copy, (x, y), 1, (0, 0, 255), 2)

    path_seg_t = cv2.rotate(path_seg, cv2.ROTATE_90_COUNTERCLOCKWISE)

    middle_lane_t = find_middle_lane_rowwise(path_seg_t)
    middle_lane_coords_t = []
    for i in range(len(middle_lane_t)):
        if middle_lane_t[i] != 0:
            middle_lane_coords_t.append((i, middle_lane_t[i]))

    middle_lane_coords_t = np.array(middle_lane_coords_t)
    # middle_lane_coords_t_restore = np.array([[0, -1], [1, 0]]) @ middle_lane_coords_t.T + np.array([[W], [0]])
    middle_lane_coords_t_restore = np.copy(middle_lane_coords_t)
    middle_lane_coords_t_restore[:, 1] = w - middle_lane_coords_t[:, 0]
    middle_lane_coords_t_restore[:, 0] = middle_lane_coords_t[:, 1]

    # for y, x in middle_lane_coords_t:
    #     cv2.circle(seg_copy, (np.abs(y-W), x), 1, (255, 0, 0), 2)

    for y, x in middle_lane_coords_t_restore:
        cv2.circle(seg_copy, (x, y), 1, (255, 0, 0), 2)

    # for coord in middle_lane_coords_t_restore:
    #     cv2.circle(seg_copy, (coord[0], coord[1]), 1, (128, 128, 128), 2)

    # middle_lane_coords_t = np.array(middle_lane_coords_t)
    # func_middle_lane_t_coef = np.polyfit(middle_lane_coords_t_restore[:, 1], middle_lane_coords_t_restore[:, 0], 3)
    func_middle_lane_t_coef = np.polyfit(middle_lane_coords_t_restore[:, 1], middle_lane_coords_t_restore[:, 0], 5)

    for x in range(w):
        y = int(fit_polynomials_scalar(x, func_middle_lane_t_coef))
        cv2.circle(seg_copy, (x, y), 1, (128, 128, 128), 2)

    # # restore to the same coordinate system
    
    min_1_x = np.min(middle_lane_coords[:, 1])
    max_1_x = np.max(middle_lane_coords[:, 1])

    min_2_x = np.min(middle_lane_coords_t_restore[:, 1])
    max_2_x = np.max(middle_lane_coords_t_restore[:, 1])

    domain_x = range(max(min_1_x, min_2_x), max(max_1_x, max_2_x))
    print(min_1_x, max_1_x, min_2_x, max_2_x)
    # initial_guess = [1, 1, 1, 1]
    
    # ideal_curve = optimize.minimize(lambda x: cost_func(domain_x, x, func_middle_lane_coef, func_middle_lane_t_coef), initial_guess)
    print(func_middle_lane_coef)
    print(func_middle_lane_t_coef)
    # print(ideal_curve.x)

    # for x in range(W):
    #     y = int(fit_polynomials_scalar(x, ideal_curve.x))
    #     cv2.circle(seg_copy, (x, y), 1, (255, 192, 203), 2)
    poly1 = np.poly1d(func_middle_lane_coef)
    poly2 = np.poly1d(func_middle_lane_t_coef)

    y1 = poly1(domain_x)
    y2 = poly2(domain_x)

    y_avg = (y1 + y2) / 2

    ideal_curve = interp1d(domain_x, y_avg, kind="cubic")
    for x in domain_x:
        y = int(ideal_curve(x))
        cv2.circle(seg_copy, (x, y), 1, (255, 192, 203), 2)

    return seg_copy

def thinning_method(binary_mask, method="zhang"):
    if len(binary_mask.shape) == 3:
        binary_mask = binary_mask[:,:,0]
    if method == "guo":
        method_ = cv2.ximgproc.THINNING_GUOHALL
    elif method == "zhang":
        method_ = cv2.ximgproc.THINNING_ZHANGSUEN
    else:
        raise ValueError(f"{method} is not implemented. Please choose `guo` | `zhang` instead.")

    skeleton = cv2.ximgproc.thinning(binary_mask, thinningType=method_)
    c, _ = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    points = None
    if len(c) != 0:
        points = c[0]
        points = points.squeeze(1)
    return points

def mean_std_dev(points):
    points_array = np.array(points)
    mean = np.mean(points_array, axis=0)
    std_dev = np.std(points_array, axis=0)
    return mean, std_dev

def filter_outliers(points, mean, std_dev, threshold=2.0):
    # Convert to numpy array for easier manipulation
    points_array = np.array(points)
    # Calculate Z-scores
    z_scores = np.abs((points_array - mean) / std_dev)
    # Filter out points where the z-score is below the threshold
    filtered_indices = np.all(z_scores < threshold, axis=1)
    filtered_points = points_array[filtered_indices]
    return list(map(tuple, filtered_points))

def sin_cos_composition_func(x, A1, B1, C1, A2, B2, C2, D):
    return A1 * np.sin(B1 * x + C1) + A2 * np.cos(B2 * x + C2) + D

def norm_x(x, x_min, x_max):
    x_normalized = 2 * np.pi * (x - x_min) / (x_max - x_min)
    return x_normalized

def denorm_x(x_normalized, x_min, x_max):
    return x_min + (x_max - x_min) * x_normalized / (2 * np.pi)

def sin_cos_composition_fit(x, y):
    x_min = np.min(x)
    x_max = np.max(x)

    x_normalized = norm_x(x, x_min, x_max)

    initial_guess = [3, 2, 0.5, 1.5, 1, 1, 4]
    params, _ = optimize.curve_fit(sin_cos_composition_func, x_normalized, y, p0=initial_guess)
    return params, x_normalized, x_min, x_max

def classify_by_gradients(binary_mask):
    h, w = binary_mask.shape
    left_mark_sparse_coord = []
    right_mark_sparse_coord = []

    for i in range(h):
        row = binary_mask[i, :]
        found_left_edge = False
        for j in range(1, w):
            # Detect left edge
            if not found_left_edge and row[j] > row[j-1]:
                left_mark_sparse_coord.append((i, j))
                found_left_edge = True
            # Detect right edge
            elif found_left_edge and row[j] < row[j-1]:
                right_mark_sparse_coord.append((i, j))
                break  # Optional: remove to find multiple edges per row

    return left_mark_sparse_coord, right_mark_sparse_coord

def transform_coordinates_rot90_clockwise(coordinates_t, origianl_w, return_type=None):
    """coordinates are in [y, x] format
    """
    coordinates_t_array = np.array(coordinates_t)
    coords_t_restore = np.copy(coordinates_t_array)
    coords_t_restore[:, 1] = origianl_w - coordinates_t_array[:, 0]
    coords_t_restore[:, 0] = coordinates_t_array[:, 1]
    if return_type is None:
        return coords_t_restore

    elif return_type == "list":
        return list(coords_t_restore)
    
    else:
        raise ValueError(f"Unsupported return type: {return_type}")

def get_left_and_right_sparse_coordinates(binary_mask, is_filtering:bool=False):
    radius = calculate_kernel_radius(binary_mask)
    gaus = gaussian_blur(binary_mask, [radius, radius])
    detector = np.ones((1, radius), np.uint8)
    conv = convolve2d(gaus, detector, 'same', 'symm')
    filtered_conv = np.zeros_like(conv, dtype=np.uint8)
    filtered_conv[conv == 255] = 255

    left_mark_sparse_coord = []
    right_mark_sparse_coord = []

    chosen_indices = np.where(conv == 255)
    list_chosen_indices = []
    for i in range(len(chosen_indices[0])):
        list_chosen_indices.append(np.array([chosen_indices[0][i], chosen_indices[1][i]]))
    map_indices = construct_map(list_chosen_indices)

    min_y = np.min(chosen_indices[0])
    max_y = np.max(chosen_indices[0])

    sep = (max_y - min_y) // 10
    begin_row = min_y
    horizontal_sum = 0
    recorder = []
    count = 0
    for k in map_indices.keys():
        if k <= begin_row + sep:
            horizontal_sum += np.mean(map_indices[k])
            count += 1
        else:
            recorder.append([(begin_row, k-1), horizontal_sum / count])
            begin_row = k
            horizontal_sum = 0
            count = 0

    # put label by this virtual middle points
    for key, value in map_indices.items():
        for info in recorder:
            range_info = info[0]
            thr = info[1]
            if key >= range_info[0] and key <= range_info[1]:
                for v in value:
                    if v <= thr:
                        left_mark_sparse_coord.append((key, v))
                    else:
                        right_mark_sparse_coord.append((key, v))

    # TODO: do we have better outlier detection algorithm?
    if is_filtering:
        left_mean, left_std_dev = mean_std_dev(left_mark_sparse_coord)
        left_mark_sparse_coord = filter_outliers(left_mark_sparse_coord, left_mean, left_std_dev)

        right_mean, right_std_dev = mean_std_dev(right_mark_sparse_coord)
        right_mark_sparse_coord = filter_outliers(right_mark_sparse_coord, right_mean, right_std_dev)

    return left_mark_sparse_coord, right_mark_sparse_coord

def classify_by_svm(binary_mask, show_plot:False):
    left_mark_sparse_coord, right_mark_sparse_coord = get_left_and_right_sparse_coordinates(binary_mask)

    data = np.zeros((len(left_mark_sparse_coord) + len(right_mark_sparse_coord), 3))
    data[0:len(left_mark_sparse_coord), 0:2] = np.array(left_mark_sparse_coord)
    data[0:len(left_mark_sparse_coord), 2] = 0
    data[len(left_mark_sparse_coord):, 0:2] = np.array(right_mark_sparse_coord)
    data[len(left_mark_sparse_coord):, 2] = 1

    X = data[:, 0:2]
    Y = data[:, 2]
    svm_clf = svm.NuSVC(kernel='rbf', gamma='auto', class_weight='balanced', verbose=False)
    svm_clf.fit(X, Y)
    
    
    if show_plot:
        x_min = X[:,1].min()
        x_max = X[:,1].max()
        y_min = X[:,0].min()
        y_max = X[:,0].max()

        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                        np.linspace(y_min, y_max, 500))
        Z = svm_clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.imshow(
        Z,
        interpolation="nearest",
        extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        aspect="auto",
        origin="lower",
        cmap=plt.cm.PuOr_r,
        )
        contours = plt.contour(xx, yy, Z, levels=[0], linewidths=2, linestyles="dashed")
        plt.scatter(X[:, 1], X[:, 0], s=30, c=Y, cmap=plt.cm.Paired, edgecolors="k")
        plt.xticks(())
        plt.yticks(())
        plt.title('SVM Decision Boundary with RBF Kernel')
        plt.show()

    return svm_clf

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):

    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    # 3) Take the absolute value of the derivative or gradient
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))

    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255.*abs_sobel/np.max(abs_sobel))

    # 5) Create a mask of 1's where the scaled gradient magnitude
    # is > thresh_min and < thresh_max
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return binary_output

def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):

    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # 3) Calculate the magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)

    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)

    # 5) Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 1

    return binary_output

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    """ threshold according to the direction of the gradient

    :param img:
    :param sobel_kernel:
    :param thresh:
    :return:
    """

    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # 3) Take the absolute value of the x and y gradients
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))

    # 5) Create a binary mask where direction thresholds are met
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    return binary_output

def gradient_pipeline(image, ksize = 3, sx_thresh=(20, 100), sy_thresh=(20, 100), m_thresh=(30, 100), dir_thresh=(0.7, 1.3)):

    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=sx_thresh)
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=sy_thresh)
    mag_binary = mag_thresh(image, sobel_kernel=ksize, thresh=m_thresh)
    dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=dir_thresh)
    combined = np.zeros_like(mag_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    # combined[(gradx == 1)  | ((mag_binary == 1) & (dir_binary == 1))] = 1
    return combined


if __name__ == "__main__":
    test_img = "/home/william/extdisk/data/cybathlon/footpath_bag_data/ai_seg_0102/1703138025.638516_result.jpg"
    img = cv2.imread(test_img)
    kernel = np.ones((11, 11), np.uint8)
    eroded = cv2.erode(img, kernel, iterations=1)
    dilated = cv2.dilate(eroded, kernel, iterations=1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1.5)
    thr = 127
    res = cv2.threshold(blur, thr, 255, cv2.THRESH_BINARY)


    cv2.imshow("original", img)
    cv2.imshow("gaussian", res[1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # rootdir = "/home/william/data/cybathlon/cppresult/"

    # all_filenames = get_filenames(rootdir, "mask")
    # # sourcedir = "/home/william/extdisk/Data/cybathlon" 
    # targetdir = "/home/william/data/cybathlon"
    # outdir = os.path.join(targetdir, "thining_method_zhang_TC89_KCOS")
    # os.makedirs(outdir, exist_ok=True)
    # for path in all_filenames:
    #     path_seg = cv2.imread(path)
    #     path_seg = cv2.GaussianBlur(path_seg, [3, 3], cv2.BORDER_DEFAULT)
    #     # coordinates = extract_boundary_line(path_seg)
    #     # optimized_middle_lane = calculate_middle_lane(path_seg, coordinates)
    #     coordinates = thining_method(path_seg, "zhang")
    #     # middle_lane = find_middle_lane_rowwise(path_seg)
    #     # middle_lane_coords = []
    #     # for i in range(len(middle_lane)):
    #     #     if middle_lane[i] != 0:
    #     #         middle_lane_coords.append((i, middle_lane[i]))
    #     if coordinates is not None:
    #         seg_copy = np.copy(path_seg)
    #         # for y, x in middle_lane_coords:
    #         for x, y in coordinates:
    #             cv2.circle(seg_copy, (x, y), 1, (255, 192, 203), 2)
    #         match = re.search(r'(\d+)_mask\.jpg', path)
    #         cv2.imwrite(os.path.join(outdir, f"{match.group(1)}_center_line.jpg"), seg_copy[:,:,::-1])
    
    # print("done!")



    # img_path = "output/sam_footpath_1702370792.414842.jpg"
    # # img_path = "data/000145_mask.jpg"
    # # img_path = "data/000145_mask.jpg"
    # path_seg = cv2.imread(img_path)
    # # show_pic(path_seg)
    # # left_coords, right_coords = get_left_and_right_sparse_coordinates(path_seg[:,:,0])
    # path_seg_t = cv2.rotate(path_seg, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # left_coords_t, right_coords_t = get_left_and_right_sparse_coordinates(path_seg_t[:,:,0])
    # left_coords = transform_coordinates_rot90_clockwise(left_coords_t, path_seg_t.shape[0], "list")
    # right_coords = transform_coordinates_rot90_clockwise(right_coords_t, path_seg_t.shape[0], "list")
    # # seg_copy = np.copy(path_seg)
    # blank_paint = np.zeros_like(path_seg, dtype=np.uint8)
    # for y, x in left_coords:
    #     cv2.circle(blank_paint, (x, y), 1, (255, 0, 0), 1)
    # for y, x in right_coords:
    #     cv2.circle(blank_paint, (x, y), 1, (0, 0, 255), 1)
    # path_seg = cv2.GaussianBlur(path_seg, [radius, radius], cv2.BORDER_DEFAULT)

    # skeleton_coords = thinning_method(path_seg[:,:,0])
    # mean, std_dev = mean_std_dev(skeleton_coords)
    # new_skeleton_coords = filter_outliers(skeleton_coords, mean, std_dev)
    # new_skeleton_coords_array = np.array(new_skeleton_coords)
    # coef = np.polyfit(new_skeleton_coords_array[:, 0], new_skeleton_coords_array[:, 1], 3)
    # print(coef)

    # # skeleton_coords = find_middle_lane_rowwise(path_seg[:,:,0]) 
    # for x, y in new_skeleton_coords:
    #     cv2.circle(seg_copy, (x, y), 1, (255, 192, 203), 1)

    # # for x in range(np.min(new_skeleton_coords_array[:, 0]), np.max(new_skeleton_coords_array[:, 0])):
    # #     y = int(fit_polynomials_scalar(x, coef))
    # #     cv2.circle(seg_copy, (x, y), 1, (255, 0, 0), 1)
        
    # # params, x_normalized, x_min, x_max = sin_cos_composition_fit(new_skeleton_coords_array[:, 0], new_skeleton_coords_array[:, 1])
    
    # # y_fit = sin_cos_composition_func(x_normalized, *params).astype(np.uint8)
    # # x_restore = denorm_x(x_normalized, x_min, x_max).astype(np.uint8)


    # # for i, j in zip(x_restore, y_fit):
    # #     cv2.circle(seg_copy, (i, j), 1, (0, 0, 255), 1)



