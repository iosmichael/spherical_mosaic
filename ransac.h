#pragma once

#include <iostream>
#include <vector>
#include <opencv/cv.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;

float minCost = 1e4;

void RANSAC(Mat &K, std::vector<DMatch> &matches, std::vector<KeyPoint> &kpt1, std::vector<KeyPoint> &kpt2);

void minimumSolver(Mat &K, Mat &R);

// construct homography and compute the reprojection error
void computeReprojectError(Mat &K, Mat &R, std::vector<KeyPoint> &kpt1, std::vector<KeyPoint> &kpt2);

void Homogenize(Mat &kp);

void Dehomogenize(Mat &kp);