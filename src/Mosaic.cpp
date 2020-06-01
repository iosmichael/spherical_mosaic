#include "Mosaic.h"
#include "Utility.h"

MosaicCamera::MosaicCamera(float fov, int width, int height, cv::Mat refRotation): fov(fov), width(width), height(height), refRotation(refRotation) { 
    using std::tan;

    const float fx = ( width  / static_cast<float>( 2 ) ) / tan( fov / 2 );
    const float fy = ( height / static_cast<float>( 2 ) ) / tan( fov / 2 );
    const float cx = ( width  - 1 ) / static_cast<float>( 2 );
    const float cy = ( height - 1 ) / static_cast<float>( 2 );
    virtualK = cv::Mat::eye(cv::Size(3,3), CV_32F);
    virtualK.at<float>(0,0) = fx;
    virtualK.at<float>(1,1) = fy;
    virtualK.at<float>(0,2) = cx;
    virtualK.at<float>(1,2) = cy;
}

MosaicCamera::~MosaicCamera() { }

void MosaicCamera::Visualize(cv::Mat refRotation, std::vector<Frame *>frames) {
    
    cv::Mat mosaic;

    float frameWidth = 2000, frameHeight = 2000;

    std::vector<Frame *> inlierFrames;

    for (auto f : frames) {
        cv::Mat R = refRotation * f->R;
        std::vector<float> limits = Utility::MosaicLimits(R, f->K, cv::Size(f->img.cols, f->img.rows));
        // test if the frame is inside the view
        float xMin = limits[0], yMin = limits[1], xMax = limits[2], yMax = limits[3];
        // find the frame size here
        if ((xMax > -10 && yMax > -10 && xMin > -10 && yMin > -10)) {
            std::cout << "including frame " << f->frameId << std::endl;
            std::cout << "\ttop left coord: " << xMin << ", " << yMin << std::endl;
            std::cout << "\tbottom right coord: " << xMax << ", " << yMax << std::endl;
            inlierFrames.push_back(f);
        }
    }

    mosaic = cv::Mat::zeros(cv::Size(frameHeight, frameWidth), CV_32F); 

    for (auto f: inlierFrames) {
        std::cout << "stitching frame " << f->frameId << " in the mosaic reconstruction" << std::endl;
        cv::Mat R = refRotation * f->R;
        Utility::SphericalWarp(f->img, R, f->K, f->K, mosaic);
    }

    std::cout << "Visualizing Mosaic Reconstruction" << std::endl;
    cv::imshow("Display Mosaic Reconstruction", mosaic);
    cv::waitKey(0);
    cv::imwrite("../demo/pitch0.png", mosaic);
}