cmake_minimum_required(VERSION 3.5.1)
project(calculate_skeleton)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

file(GLOB sources ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp)
file(GLOB headers ${CMAKE_CURRENT_LIST_DIR}/src/*.h)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenMP)

add_subdirectory(3rdparty/libsvm)
message(STATUS ${svm_include_dir})

set(mandatory_libs ${OpenCV_LIBS} svm)

if(OpenMP_CXX_FOUND)
    list(APPEND {mandatory_libs} OpenMP::OpenMP_CXX)
endif()

add_executable(${PROJECT_NAME} 
        main.cpp
        ${sources}
)

target_include_directories(
        ${PROJECT_NAME}
        PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        ${svm_include_dir}
)

target_link_libraries(
        ${PROJECT_NAME}
        ${mandatory_libs}
)