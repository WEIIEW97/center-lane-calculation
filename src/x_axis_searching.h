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

#ifndef CALCULATE_SKELETON_X_AXIS_SEARCHING_H
#define CALCULATE_SKELETON_X_AXIS_SEARCHING_H

#include "common.h"
#include "opencv2/opencv.hpp"
namespace clc {
  std::vector<cv::Point2i> search_x_axis(const cv::Mat& binary_mask);
}

#endif // CALCULATE_SKELETON_X_AXIS_SEARCHING_H
