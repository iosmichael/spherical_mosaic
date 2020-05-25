#pragma once
#include <map>
#include <opencv2/opencv.hpp>
#include "Frame.h"

class Point {
public:
    int pointID;
    cv::Point3f pt;
    std::map<Frame *, int> observations;
    
    Point(int id);
    ~Point(); 
    
    void AddObservation(Frame *frame, int index);
private:
    void Update();
};