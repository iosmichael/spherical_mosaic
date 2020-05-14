#pragma once

#include <opencv/cv.hpp>
#include <eigen3/Eigen/Core>

namespace Mosaic {

class Frame {
    public:
    cv::Mat image;

    // Image, Camera Pose (to reference frame), K and Kinv, id
    // features and feature indices, normalized feature points

}

class Feature2D {
    public:
    // frame idx, uv_coords, SIFT_descriptors
}

}