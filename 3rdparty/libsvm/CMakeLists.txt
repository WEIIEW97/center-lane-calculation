file(GLOB svm_sources ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
set(svm_include_dir ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE)
add_library(svm ${svm_sources})