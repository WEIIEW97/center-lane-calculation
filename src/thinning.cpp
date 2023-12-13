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
#include "thinning.h"
#include "common.h"

namespace clc {
  void algo_zhangsuen_impl(cv::Mat img, cv::Mat& marker, int iter) {
    for (int i = 1; i < img.rows - 1; i++) {
      for (int j = 1; j < img.cols - 1; j++) {
        uchar p2 = img.at<uchar>(i - 1, j);
        uchar p3 = img.at<uchar>(i - 1, j + 1);
        uchar p4 = img.at<uchar>(i, j + 1);
        uchar p5 = img.at<uchar>(i + 1, j + 1);
        uchar p6 = img.at<uchar>(i + 1, j);
        uchar p7 = img.at<uchar>(i + 1, j - 1);
        uchar p8 = img.at<uchar>(i, j - 1);
        uchar p9 = img.at<uchar>(i - 1, j - 1);

        int A = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
                (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
                (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
                (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
        int B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
        int m1 = iter == 0 ? (p2 * p4 * p6) : (p2 * p4 * p8);
        int m2 = iter == 0 ? (p4 * p6 * p8) : (p2 * p6 * p8);

        if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0)
          marker.at<uchar>(i, j) = 1;
      }
    }
  }

  void algo_guohall_impl(cv::Mat img, cv::Mat& marker, int iter) {
    for (int i = 1; i < img.rows - 1; i++) {
      for (int j = 1; j < img.cols - 1; j++) {
        uchar p2 = img.at<uchar>(i - 1, j);
        uchar p3 = img.at<uchar>(i - 1, j + 1);
        uchar p4 = img.at<uchar>(i, j + 1);
        uchar p5 = img.at<uchar>(i + 1, j + 1);
        uchar p6 = img.at<uchar>(i + 1, j);
        uchar p7 = img.at<uchar>(i + 1, j - 1);
        uchar p8 = img.at<uchar>(i, j - 1);
        uchar p9 = img.at<uchar>(i - 1, j - 1);

        int C = ((!p2) & (p3 | p4)) + ((!p4) & (p5 | p6)) +
                ((!p6) & (p7 | p8)) + ((!p8) & (p9 | p2));
        int N1 = (p9 | p2) + (p3 | p4) + (p5 | p6) + (p7 | p8);
        int N2 = (p2 | p3) + (p4 | p5) + (p6 | p7) + (p8 | p9);
        int N = N1 < N2 ? N1 : N2;
        int m = iter == 0 ? ((p6 | p7 | (!p9)) & p8) : ((p2 | p3 | (!p5)) & p4);

        if ((C == 1) && ((N >= 2) && ((N <= 3)) & (m == 0)))
          marker.at<uchar>(i, j) = 1;
      }
    }
  }

  void thinning_iteration(cv::Mat& img, int iter, int method) {
    cv::Mat marker = cv::Mat::zeros(img.size(), CV_8UC1);

    if (method == ZHANGSUEN) {
      algo_zhangsuen_impl(img, marker, iter);
    }

    if (method == GUOHALL) {
      algo_guohall_impl(img, marker, iter);
    }

    img &= ~marker;
  }

  // void thinning(const cv::Mat& src, cv::Mat dst, int method) {
  //   cv::Mat processed = src.clone();
  //   CV_CheckTypeEQ(processed.type(), CV_8UC1, "");

  //   processed /= 255;

  //   cv::Mat prev, diff;
  //   prev = cv::Mat::zeros(processed.size(), CV_8UC1);

  //   do {
  //     thinning_iteration(processed, 0, method);
  //     thinning_iteration(processed, 1, method);
  //     cv::absdiff(processed, prev, diff);
  //     processed.copyTo(prev);
  //   } while (cv::countNonZero(diff) > 0);

  //   processed *= 255;
  //   dst = std::move(processed);
  // }
  void thinning(cv::InputArray input,cv::OutputArray output, int thinningType){
    cv::Mat processed = input.getMat().clone();
    CV_CheckTypeEQ(processed.type(), CV_8UC1, "");
    // Enforce the range of the input image to be in between 0 - 255
    processed /= 255;

    cv::Mat prev =cv:: Mat::zeros(processed.size(), CV_8UC1);
    cv::Mat diff;

    do {
        thinning_iteration(processed, 0, thinningType);
        thinning_iteration(processed, 1, thinningType);
        absdiff(processed, prev, diff);
        processed.copyTo(prev);
    }
    while (countNonZero(diff) > 0);

    processed *= 255;

    output.assign(processed);
}
} // namespace clc