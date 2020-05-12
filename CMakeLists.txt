cmake_minimum_required(VERSION 3.0.0)
project(mosaic VERSION 0.1.0)

include(CTest)
enable_testing()

add_subdirectory(glog)

find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
find_package(Eigen3 REQUIRED)

add_executable(mosaic main.cpp)

include_directories(${OpenCV_INCLUDE_DIRS} ${CERES_INCLUDE_DIR} ${EIGEN3_INCLUDE_DIRS} ${glog_INCLUDE_DIR})

target_link_libraries(mosaic ${OpenCV_LIBS} ${CERES_LIBRARIES} Eigen3::Eigen glog::glog)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
