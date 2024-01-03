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
#include <string>
#include "src/methods.h"
#include "src/spline.h"

int main() {
    std::string img_path = "/home/william/extdisk/data/cybathlon/footpath_bag_data/ai_seg_0102/1703138025.638516_result.jpg";
    cv::Mat path_seg = cv::imread(img_path);
    cv::Mat seg_copy = path_seg.clone();
    int h = path_seg.rows;
    int w = path_seg.cols;
    std::vector<cv::Mat> channels;
    cv::split(path_seg, channels);

    auto skeleton_coords = clc::row_searching_reduce_method(channels[0]);
    std::vector<double> X, Y;
    for (auto& p : skeleton_coords) {
      cv::circle(seg_copy, p, 1, cv::Scalar(255, 192, 203), 2);
      X.push_back(p.x);
      Y.push_back(p.y);
    }

    tk::spline s(Y, X);
    std::vector<cv::Point> spline_coords;
    for (int i = 0; i < Y.size(); i++) {
      int original_y = Y[i];
      int fitted_x = s(original_y);
      std::cout << "diff: " << X[i] - fitted_x << std::endl;
      spline_coords.emplace_back(fitted_x, original_y);
    }
    for (auto& p : spline_coords) {
      cv::circle(seg_copy, p, 1, cv::Scalar(255, 0, 0), 2);
    }

//    cv::imshow("middle lane", seg_copy);
//    cv::waitKey(0);
//    cv::destroyAllWindows();
    return 0;
}