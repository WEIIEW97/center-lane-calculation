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

#ifndef CALCULATE_SKELETON_TICTOC_H
#define CALCULATE_SKELETON_TICTOC_H

#include <chrono>
#include <iostream>

#define TIC(id)                                                                \
  auto start_time_##id = std::chrono::high_resolution_clock::now();
#define TOC(id)                                                                \
  auto end_time_##id = std::chrono::high_resolution_clock::now();              \
  auto time_elapsed_##id = end_time_##id - start_time_##id;                    \
  std::cout                                                                    \
      << "Time elapsed for Tic-Toc " << #id << ": "                            \
      << std::chrono::duration<double, std::milli>(time_elapsed_##id).count()  \
      << " ms" << std::endl;

#endif // CALCULATE_SKELETON_TICTOC_H
