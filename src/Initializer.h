#pragma once
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv/cv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/highgui.hpp>
#include "Frame.h"

class Initializer {
public:
    // static variables
    cv::Ptr<cv::xfeatures2d::SIFT> detector = cv::xfeatures2d::SIFT::create();
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
    void initRotation();
};