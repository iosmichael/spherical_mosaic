#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <tuple>

namespace Utility
{
    inline cv::Mat Homogenize(cv::Mat X) {
        // X should have the size of 2 x N
        X.push_back(cv::Mat::ones(1, X.cols, CV_32F));
        assert(X.rows == 3);
        return X;
    }

    inline cv::Mat Dehomogenize(cv::Mat X) {
        // X should have the size of 3 x N
        for (size_t i = 0; i < X.cols; i++) {
            X.at<float>(0, i) = X.at<float>(0, i) / X.at<float>(2, i);
            X.at<float>(1, i) = X.at<float>(1, i) / X.at<float>(2, i);
        }
        // return size should be of 2 x N
        return X.rowRange(cv::Range(0,2));
    }
    
    inline cv::Mat Normalize(cv::Mat X, cv::Mat K) {
        // X is the homogeneous vector of size 3 x N
        cv::Mat normalized = K.inv() * X;
        for (size_t i = 0; i < normalized.cols; i++) {
            // calculate the norm of vector column
            double norm = cv::norm(normalized.col(i), cv::NORM_L2);
            // calculate the sign of the third coord: sign(z)
            int sign = (normalized.at<float>(2, i) < 0) ? -1 : 1;
            norm = norm * sign;
            normalized.at<float>(0, i) /= norm;
            normalized.at<float>(1, i) /= norm;
            normalized.at<float>(2, i) /= norm;
        }
        return normalized;
    }

    // min_x, min_y, max_x, max_y
    inline std::vector<float> MosaicLimits(cv::Mat &R, cv::Mat &K, cv::Size size) {
        cv::Mat H = K * R * K.inv();
        cv::Mat origin = cv::Mat::zeros(3,1,CV_32F), bottomRight = (cv::Mat_<float>(3,1) << size.width, size.height, 1);
        origin = Dehomogenize(H * origin), bottomRight = Dehomogenize(H * bottomRight);
        std::vector<float> limits;
        float xMin = origin.at<float>(0,0), yMin = origin.at<float>(1,0);
        float xMax = bottomRight.at<float>(0,0), yMax = bottomRight.at<float>(1,0);
        return std::vector<float> {xMin, yMin, xMax, yMax};
    }

    void WarpLocal(cv::Mat &src, std::pair<cv::Mat, cv::Mat> &xyMap, cv::Mat &dst);

    void SphericalWarp(cv::Mat &src, cv::Mat &R, cv::Mat &vK, cv::Mat &K, cv::Mat &dst);

    void ComputeSphericalWarpMappings(cv::Mat &img, cv::Mat &R, cv::Mat &vK, cv::Mat &K, std::pair<cv::Mat, cv::Mat> &xyMap);

    void PanoramaSphericalWarp(cv::Mat img);

} // namespace Utility
