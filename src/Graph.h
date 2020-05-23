/*
Graph Class
- Point and Frame Relationship
- Bundle Adjustment with Ceres

author: Michael, Suhrid
date: 5/23/2020
*/

#pragma once
#include <vector>
#include "Frame.h"
#include "Point.h"

class Graph {
public:
    std::vector<Frame *> frames;
    std::vector<Point *> points;

    Graph();
    ~Graph();

    void AddFrame(cv::Mat img);
    
private:
    void Optimize();
};