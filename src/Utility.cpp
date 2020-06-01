#include "Utility.h"

void Utility::SphericalWarp(cv::Mat &src, cv::Mat &R, cv::Mat &vK, cv::Mat &K, cv::Mat &dst) {
    std::pair<cv::Mat, cv::Mat> xyMaps;
    ComputeSphericalWarpMappings(src, R, vK, K, xyMaps);
    WarpLocal(src, xyMaps, dst);
}

void Utility::ComputeSphericalWarpMappings(cv::Mat &img, cv::Mat &R, cv::Mat &vK, cv::Mat &K, std::pair<cv::Mat, cv::Mat> &xyMap) { 
    int height = img.rows, width = img.cols;
    float fx = vK.at<float>(0,0), fy = vK.at<float>(1,1);
    float cx = vK.at<float>(0,2), cy = vK.at<float>(1,2);
    xyMap.first = cv::Mat::zeros(height, width, CV_32F);
    xyMap.second = cv::Mat::zeros(height, width, CV_32F);

    for (size_t i = 0; i < height; i++) {
        for (size_t j = 0; j < width; j++) {
            // perform inverse spherical projection: uv -> XYZ
            // std::cout << "uv coord " << i << ", " << j << std::endl;
            float xf = (j - cx) / fx, yf = (i - cy) / fy;
            cv::Mat X = (cv::Mat_<float>(3,1,CV_32F) << std::sin(xf) * std::cos(yf), std::sin(yf), std::cos(xf) * std::cos(yf));
            // cv::Mat X = (cv::Mat_<float>(3,1,CV_32F) << j, i, 1);
            // perform forward spherical projection: XYZ->uv
            X = Utility::Dehomogenize(K * R.inv() * X);
            xyMap.first.at<float>(i, j) = X.at<float>(0,0);
            xyMap.second.at<float>(i, j) = X.at<float>(1,0);
        }
    }
}

void Utility::WarpLocal(cv::Mat &src, std::pair<cv::Mat, cv::Mat> &xyMap, cv::Mat &dst) {
    float height = src.rows, width = src.cols;
    cv::Mat mask = cv::Mat::ones(height, width, CV_32F);
    cv::inRange(xyMap.second, 0, height-1, mask);
    cv::inRange(xyMap.first, 0, width-1, mask);

    cv::Mat warpImg;
    cv::remap(src, warpImg, xyMap.first, xyMap.second, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    cv::bitwise_and(warpImg, warpImg, dst, mask);
}

void Utility::PanoramaSphericalWarp(cv::Mat img) {
    cv::Mat panorama;
    cv::Mat K = cv::Mat::zeros(3, 3, CV_32F);
    cv::Mat R = cv::Mat::eye(3, 3, CV_32F);
    cv::detail::SphericalWarper warper {1.0f};
    warper.warp(img, K, R, cv::INTER_LINEAR, cv::BORDER_DEFAULT, panorama);
    std::cout << "Image Spherical Warpping" << std::endl;
    
    // cv::imshow("Spherical Panorama", panorama);
    // cv::waitKey(0);
}