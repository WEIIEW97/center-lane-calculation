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

#ifndef CALCULATE_SKELETON_PREPROCESS_H
#define CALCULATE_SKELETON_PREPROCESS_H

#include "common.h"
#include <opencv2/opencv.hpp>

namespace clc {
  void gaussian_blur(const cv::Mat& img, cv::Mat& dst, int kernel_size[2]);
  std::vector<cv::Point2i> extract_boundary_line(const cv::Mat& binary_mask,
                                                 int thr = 250);
} // namespace clc

#endif // CALCULATE_SKELETON_PREPROCESS_H
