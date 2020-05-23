/*
Initializer Class
- SIFT Detection/Descriptor Algorithm
- KNN BFMatcher Matching Algorithm
- RANSAC + Minimum Solver (Calibrated Rotation Matrix)
- Outlier Rejection
- ScenePoint Initialization

author: Michael, Suhrid
date: 5/23/2020
*/


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
    float ComputeCost(cv::Mat &R, float tolerance, std::vector<int> &inliers);
    void ComputeInliers(cv::Mat &R);
    void SolveCalibratedRotation(std::vector<cv::DMatch> matches, cv::Mat &solution);
};