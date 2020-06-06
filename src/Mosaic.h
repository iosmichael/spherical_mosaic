#pragma once

#include <opencv2/opencv.hpp>
#include "Frame.h"

class MosaicCamera {
public:
    // refRotation is the rotation from the first frame to the virtual frame
    MosaicCamera(float fov, int width, int height, cv::Mat refRotation);

    void Visualize(cv::Mat refRotation, std::vector<Frame *>frames);

    void Panorama(cv::Mat refRotation, std::vector<Frame *>frames);

    ~MosaicCamera();

private:

    float fov, width, height;
    cv::Mat refRotation;
    cv::Mat virtualK;

    cv::Mat makeMask(cv::Size sz, std::vector<cv::Point2f> imageCorners, cv::Mat H);

};