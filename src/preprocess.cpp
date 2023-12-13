/*
 * Copyright (c) 2022-2023, William Wei. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "preprocess.h"

namespace clc {
  void gaussian_blur(const cv::Mat& img, cv::Mat& dst, int kernel_size[2]) {
    cv::GaussianBlur(img, dst, cv::Size(kernel_size[0], kernel_size[1]), 0);
  }

  std::vector<cv::Point2i> extract_boundary_line(const cv::Mat& binary_mask,
                                                 int thr) {
    cv::Mat gx_right_shift, gx_left_shift, gy_down_shift, gy_up_shift;
    gx_right_shift =
        binary_mask(cv::Rect(1, 0, binary_mask.cols - 1, binary_mask.rows)) -
        binary_mask(cv::Rect(0, 0, binary_mask.cols - 1, binary_mask.rows));
    gx_left_shift =
        binary_mask(cv::Rect(0, 0, binary_mask.cols - 1, binary_mask.rows)) -
        binary_mask(cv::Rect(1, 0, binary_mask.cols - 1, binary_mask.rows));
    gy_down_shift =
        binary_mask(cv::Rect(0, 1, binary_mask.cols, binary_mask.rows - 1)) -
        binary_mask(cv::Rect(0, 0, binary_mask.cols, binary_mask.rows - 1));
    gy_up_shift =
        binary_mask(cv::Rect(0, 0, binary_mask.cols, binary_mask.rows - 1)) -
        binary_mask(cv::Rect(0, 1, binary_mask.cols, binary_mask.rows - 1));

    cv::Mat gx, gy, grads;
    cv::bitwise_or(gx_left_shift, gx_right_shift, gx);
    cv::bitwise_or(gy_up_shift, gy_down_shift, gy);

    cv::bitwise_or(gx.rowRange(1, gx.rows), gy.colRange(1, gy.cols), grads);

    int kernel_size = 3;
    cv::Mat kernel = cv::Mat::ones(kernel_size, kernel_size, CV_8U);

    cv::Mat grads_erode;
    cv::erode(grads, grads_erode, kernel, cv::Point(-1, -1), 1);

    cv::Mat valid_points = (grads_erode > thr);

    std::vector<cv::Point2i> coordinates;
    for (int i = 0; i < valid_points.rows; i++) {
      for (int j = 0; j < valid_points.cols; j++) {
        if (valid_points.at<uchar>(i, j) != 0) {
          coordinates.emplace_back(j, i);
        }
      }
    }
    return coordinates;
  }
} // namespace clc