#pragma once

#include <opencv2/opencv.hpp>
#include "Frame.h"

class Initializer {
    
    public:

    float mMinCost = 1e4;
    Frame *mFrame;

    Initializer(Frame *frame);

    void Homogenize(std::vector<cv::Mat> &pts);

    void Dehomogenize(std::vector<cv::Mat> &pts);

    void FeatureDetection();

    void FeatureMatching();

    void RANSAC(cv::Mat &K, std::vector<cv::DMatch> &matches, std::vector<cv::KeyPoint> &kpt1, std::vector<cv::KeyPoint> &kpt2);

    void ComputeMinimumRotation();

    // construct homography and compute the reprojection error
    void ComputeError(cv::Mat &K, cv::Mat &R, std::vector<cv::KeyPoint> &kpt1, std::vector<cv::KeyPoint> &kpt2);
};