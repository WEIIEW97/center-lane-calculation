/*
* Copyright (c) 2022-2023, William Wei.  All rights reserved.
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

#include <iostream>
#include "src/methods.h"

int main() {
    std::string img_path = "/home/william/Codes/center-lane-calculation/output/sam_footpath_1702370792.414842.jpg";
    cv::Mat path_seg = cv::imread(img_path);
    cv::Mat seg_copy = path_seg.clone();
    int h = path_seg.rows;
    int w = path_seg.cols;
    std::vector<cv::Mat> channels;
    cv::split(path_seg, channels);

    auto skeleton_coords = clc::row_searching_method(channels[0]);
    for (auto& p : skeleton_coords) {
      cv::circle(seg_copy, p, 1, cv::Scalar(255, 192, 203), 2);
    }

    cv::imshow("middle lane", seg_copy);
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}