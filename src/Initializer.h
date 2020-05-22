#pragma once
#include <iostream>
#include <algorithm>
#include <random>

#include <opencv2/opencv.hpp>
#include <opencv/cv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/highgui.hpp>
#include "Frame.h"

class Initializer {
public:
    // static variables
    static cv::Ptr<cv::xfeatures2d::SIFT> detector;
    const double kDistanceCoef = 4.0;
    const int kMaxMatchingSize = 50;
    // end static variables

    Frame *frame;

    Initializer(Frame *frame);
    ~Initializer();
    void initialize();

private:
    void FeatureExtractor();
    void FeatureMatcher();
    void RANSAC();
    float ComputeCost(cv::Mat &R);
    void MinimumSolverCalibratedRotation(std::vector<cv::DMatch> matches, cv::Mat &solution);
    void initRotation();
};