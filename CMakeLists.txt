cmake_minimum_required(VERSION 3.0.0)
project(mosaic VERSION 0.1.0)

include(CTest)
enable_testing()

add_subdirectory(glog)

find_package(OpenCV 3.4.2 REQUIRED)
find_package(Ceres REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem)

message("OPENCV_VERSION: " ${OpenCV_VERSION} "     [MAKE SURE IT'S 3.4.2]")
message("EIGEN3_PATH: " ${EIGEN3_INCLUDE_DIRS})

include_directories(${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS} ${CERES_INCLUDE_DIR} ${glog_INCLUDE_DIR} ${Boost_INCLUDE_DIRS})

file(GLOB MOSAIC_SOURCE_FILE "src/*.h" "src/*.cpp")

# set(CERES_SOURCE_FILE bal_test/bal_test.cpp)

add_executable(mosaic main.cpp ${MOSAIC_SOURCE_FILE})

# add_executable(mosaic ${CERES_SOURCE_FILE})

target_link_libraries(mosaic ${OpenCV_LIBS} ${CERES_LIBRARIES} Boost::filesystem Eigen3::Eigen glog::glog)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
