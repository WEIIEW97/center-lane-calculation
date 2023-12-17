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

#ifndef CALCULATE_SKELETON_METHODS_H
#define CALCULATE_SKELETON_METHODS_H

#include "opencv2/opencv.hpp"
#include "common.h"

namespace clc {
  std::vector<std::vector<cv::Point2i>>
  thinning_method(const cv::Mat& binary_mask, const std::string& method);
  std::vector<cv::Point> row_searching_method(const cv::Mat& binary_mask);
}

#endif // CALCULATE_SKELETON_METHODS_H
