/*
 * Copyright (c) 2023--present, WILLIAM WEI.  All rights reserved.
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
#include "methods.h"
#include <numeric>
#include <opencv2/ximgproc.hpp>
#include <stdexcept>
#include <omp.h>

namespace clc {
  std::vector<std::vector<cv::Point2i>>
  thinning_method(const cv::Mat& binary_mask, const std::string& method) {
    int method_;
    if (method == "guo") {
      method_ = cv::ximgproc::THINNING_GUOHALL;
    } else if (method == "zhang") {
      method_ = cv::ximgproc::THINNING_ZHANGSUEN;
    } else {
      throw std::invalid_argument(
          method +
          " is not implemented. Please choose `guo` | `zhang` instead.");
    }

    cv::Mat skeleton;
    cv::ximgproc::thinning(binary_mask, skeleton, method_);

    std::vector<std::vector<cv::Point>> contour;
    cv::findContours(skeleton, contour, cv::RETR_EXTERNAL,
                     cv::CHAIN_APPROX_SIMPLE);
    return contour;
  }

  std::vector<cv::Point> row_searching_method(const cv::Mat& binary_mask) {
    std::vector<cv::Point2i> middle_lane_coords;
    int h = binary_mask.rows;
    int w = binary_mask.cols;
    middle_lane_coords.reserve(h);

    for (int i = 0; i < h; ++i) {
      const auto* row = binary_mask.ptr<uchar>(i);
      double sum = 0;
      int count = 0;

      for (int j = 0; j < w; ++j) {
        if (row[j] == 255) {
          sum += j;
          ++count;
        }
      }

      if (count > 0) {
        double mean = sum / count;
        middle_lane_coords.emplace_back(static_cast<int>(mean), i);
      }
    }

    return middle_lane_coords;
  }

  std::vector<cv::Point>
  row_searching_reduce_method(const cv::Mat& binary_mask) {
    std::vector<cv::Point> middle_lane_coords;
    int h = binary_mask.rows;

    middle_lane_coords.reserve(h);

#pragma omp parallel for
    for (int i = 0; i < h; i++) {
      auto row = binary_mask.row(i);
      auto count = cv::countNonZero(row);

      if (count > 0) {
        cv::Mat locations;
        cv::findNonZero(row, locations);
        double sum = std::accumulate(
            locations.begin<cv::Point>(), locations.end<cv::Point>(), 0.0,
            [](double s, const cv::Point& p) { return s + p.x; });
        double mean = sum / count;
        middle_lane_coords.emplace_back(static_cast<int>(mean), i);
      }
    }
    return middle_lane_coords;
  }

  std::vector<cv::Point2f>
  spline_fitting(const std::vector<cv::Point>& points) {
    std::vector<cv::Point2f> curve;
    cv::approxPolyDP(points, curve, 1, false);
    return curve;
  }
} // namespace clc