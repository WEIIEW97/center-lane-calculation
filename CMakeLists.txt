cmake_minimum_required(VERSION 3.5.1)
project(calculate_skeleton)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

file(GLOB sources ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp)
file(GLOB headers ${CMAKE_CURRENT_LIST_DIR}/src/*.h)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)


add_executable(${PROJECT_NAME} 
        main.cpp
        ${sources}
)

target_include_directories(
        ${PROJECT_NAME}
        PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(
        ${PROJECT_NAME}
        ${OpenCV_LIBS}
)