#pragma once
#include <map>
#include <opencv2/opencv.hpp>
#include "Frame.h"

class Point {
public:
    int pointID;
    cv::Point3f pt;
    std::map<Frame *, int> observations;
    
    // points used for bundle adjustment application
    float *X = new float[3];

    Point(int id);
    ~Point(); 
    
    void AddObservation(Frame *frame, int index);
private:
    void Update();
};