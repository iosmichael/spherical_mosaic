#pragma once

#include <iostream>
#include <vector>
#include <map>
#include <tuple>
#include <string>
#include <opencv/cv.hpp>

#include "Point.h"

class Frame {
    public:
        static cv::Mat K;

        int frameId;
        bool isFirst, hasKeyPoints, isInitialize;
        Frame *refFrame;
        std::vector<cv::KeyPoint> kpts;
        std::vector<cv::DMatch> matches;
        std::vector<std::tuple<int, int>> inliers;

        // everytime we create a vector of inliers, we will check the ref->scenePts[int index]. 
        // If no scene point exists associated with the kpt index, we will create a new Point in the initializer
        // If scene point exists associated with the kpt index, we will add observation on that scene point
        std::map<int, Point *> scenePts;

        cv::Mat img, desc, refR, R;
    
        Frame(int frameId, bool isFirst, cv::Mat &imgData, Frame *refFrame);
        ~Frame();

        void visualize();
        void printStatus();
};