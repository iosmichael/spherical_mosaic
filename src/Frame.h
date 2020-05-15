#pragma once

#include <vector>
#include <opencv/cv.hpp>

class Frame {
    public:
    cv::Mat imLeft, imRight;
    cv::Mat mK;

    cv::Mat mR21;
    std::vector<cv::KeyPoint> mLeftKpts, mRightKpts;
    std::vector<cv::DMatch> mMatches;
    std::vector<bool> mbInliers;

    Frame(cv::Mat &imLeft, cv::Mat &imRight, cv::Mat &mK);

};