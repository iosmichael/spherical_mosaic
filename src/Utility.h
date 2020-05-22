#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

namespace Utility
{
    inline cv::Mat Homogenize(cv::Mat &x) {
        int col_size = x.cols;
        cv::Mat to_append = cv::Mat::ones(1, col_size, CV_32F);  // ncols cols, 1 row
        x.push_back(to_append);
        return x;
    }

    inline cv::Mat Dehomogenize(cv::Mat &x) {
        cv::Mat dehomoX, temp;
        cv::divide(x.row(0), x.row(2), temp);
        dehomoX.push_back(temp);
        cv::divide(x.row(1), x.row(2), temp);
        dehomoX.push_back(temp);
        // dehomoX is the desired output
        return dehomoX;
    }
    
    inline cv::Mat Normalize(cv::Mat &homoX, cv::Mat &K) {
        cv::Mat x_norm = K.inv() * homoX; // x_cam is obtained from camera_projection_matrix
        double norm_factor = cv::norm(x_norm, cv::NORM_L2);
        cv::divide(norm_factor, x_norm, x_norm); // divide x_norm by norm_factor and store it in x_norm
        // x = [-1, -2, -3] => [1,2,3], x = [-1,2,-1] => [1,-2,1]
        bool is_all_positive = (std::find_if(x_norm.begin<float>(), x_norm.end<float>(), [](auto x) { return (x < 0); }) == x_norm.end<float>());
        std::cout << is_all_positive << std::endl;
        if (!is_all_positive) {
            x_norm = -1*x_norm; //invert sign if all not positive
        }
        return x_norm;
    }

} // namespace Utility
