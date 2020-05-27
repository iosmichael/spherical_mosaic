#include "Mosaic.h"
#include "Utility.h"

MosaicCamera::MosaicCamera(float fov, int width, int height, cv::Mat refRotation): fov(fov), width(width), height(height), refRotation(refRotation) { }

MosaicCamera::~MosaicCamera() { }

void MosaicCamera::Visualize(cv::Mat refRotation, std::vector<Frame *>frames) {
    
    cv::Mat mosaic;

    float frameWidth = 0, frameHeight = 0;

    for (auto f : frames) {
        cv::Mat R = refRotation * f->R;
        std::vector<float> limits = Utility::MosaicLimits(R, f->K, cv::Size(f->img.rows, f->img.cols));
        // test if the frame is inside the view
        float xMin = limits[0], yMin = limits[1], xMax = limits[2], yMax = limits[3];
        // find the frame size here
        frameWidth = std::max({frameWidth, xMax});
        frameHeight = std::max({frameHeight, yMax});
    }

    mosaic = cv::Mat::zeros(cv::Size(frameHeight, frameWidth), CV_32F); 

    for (auto f: frames) {
        cv::Mat R = refRotation * f->R;
        Utility::SphericalWarp(f->img, R, f->K, mosaic);
    }

    std::cout << "Visualizing Mosaic Reconstruction" << std::endl;
    cv::imshow("Display Mosaic Reconstruction", mosaic);
    cv::waitKey(0);
}