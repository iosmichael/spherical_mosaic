#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <opencv/cv.hpp>

class Frame {
    public:
        static cv::Mat K;

        int frameId;
        bool isFirst, hasKeyPoints, isInitialize;
        Frame *refFrame;
        std::vector<cv::KeyPoint> kpts;
        std::vector<cv::DMatch> matches;
        std::vector<int> inliers;
        cv::Mat img, desc, refR, R;
    
        Frame(int frameId, bool isFirst, cv::Mat &imgData, Frame *refFrame);
        ~Frame();

        void visualize();
        void printStatus();
};