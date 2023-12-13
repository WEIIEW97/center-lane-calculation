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
#include "x_axis_searching.h"

namespace clc {
  std::vector<cv::Point2i> search_x_axis(const cv::Mat& binary_mask) {
    std::vector<cv::Point2i> middel_lane_coords;
    int h = binary_mask.rows;
    int w = binary_mask.cols;
    middel_lane_coords.reserve(h);

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
        middel_lane_coords.emplace_back(static_cast<int>(mean), i);
      }
    }

    return middel_lane_coords;
  }
} // namespace clc