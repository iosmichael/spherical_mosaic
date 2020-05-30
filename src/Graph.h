/*
Graph Class
- Point and Frame Relationship
- Bundle Adjustment with Ceres

author: Michael, Suhrid
date: 5/23/2020
*/

#pragma once
#include <vector>
#include <map>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include "Frame.h"
#include "Point.h"

class Graph {
public:
    std::vector<Frame *> frames;
    std::map<int, Point *> points;

    Graph();
    ~Graph();

    void AddFrame(cv::Mat img);
    void Optimize();
};

struct SphericalReprojectionError {
  SphericalReprojectionError(double observed_x, double observed_y, double *K)
      : observed_x(observed_x), observed_y(observed_y) {
          fx = K[0]; 
          fy = K[1];
          cx = K[2]; 
          cy = K[3];
       }

  template <typename T>
  bool operator()(const T* angle_axis,
                  const T* point,
                  T* residuals) const {
    // camera[0,1,2] are the angle-axis rotation.
    T p[3];
    ceres::AngleAxisRotatePoint(angle_axis, point, p);

    // Compute the center of distortion. The sign change comes from
    // the camera model that Noah Snavely's Bundler assumes, whereby
    // the camera coordinate system has a negative z axis.
    T xp = p[0] / p[2];
    T yp = p[1] / p[2];

    // Compute final projected point position.
    T predicted_x = fx * xp + cx;
    T predicted_y = fy * yp + cy;

    // The error is the difference between the predicted and observed position.
    residuals[0] = predicted_x - observed_x;
    residuals[1] = predicted_y - observed_y;

    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(const double observed_x,
                                     const double observed_y,
                                     double *K) {
    return (new ceres::AutoDiffCostFunction<SphericalReprojectionError, 2, 3, 3>(
                new SphericalReprojectionError(observed_x, observed_y, K)));
  }
  double observed_x;
  double observed_y;
  double fx, fy, cx, cy;
};