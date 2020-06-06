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
    
    cv::Mat mosaic, mask, mask0;

    float frameWidth = 1000, frameHeight = 1000;

    std::vector<Frame *> inlierFrames;

    cv::detail::MultiBandBlender blender;

    for (auto f : frames) {
        cv::Mat R = refRotation * f->R;
        frameWidth = f->img.size().width;
        frameHeight = f->img.size().height;
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
    int minX, minY;
    blender.prepare(cv::Rect(0,0,frameWidth, frameHeight));

    int n_blend = 2;
    for (auto f: inlierFrames) {

        std::cout << "stitching frame " << f->frameId << " in the mosaic reconstruction" << std::endl;
        cv::Mat R = refRotation * f->R;
        
        std::vector<float> limits = Utility::MosaicLimits(R, f->K, cv::Size(f->img.cols, f->img.rows));
        minX = std::min(limits[0], limits[2]);
        minY = std::min(limits[1], limits[3]);
        std::cout << minX << ", " << minY << std::endl;
        mosaic = cv::Mat::zeros(cv::Size(frameWidth, frameHeight), CV_32F); 
        mask = cv::Mat::ones(cv::Size(f->img.size().width, f->img.size().height), CV_8UC1) * 255;
        mask0 = cv::Mat::zeros(cv::Size(frameWidth, frameHeight), CV_8UC1);
        Utility::SphericalWarp(f->img, R, f->K, f->K, mosaic);
        // cv::imshow("Warped Image", mosaic);
        // cv::waitKey(0);
        Utility::SphericalWarp(mask, R, f->K, f->K, mask0);
        // cv::imshow("Warped Mask", mask0);
        // cv::waitKey(0);
        blender.feed(mosaic, mask0, cv::Point2f(0,0));
    }

    blender.blend(mosaic, mask);
    mosaic.convertTo(mosaic, (mosaic.type() / 8) * 8);
    std::cout << "Visualizing Mosaic Reconstruction" << std::endl;
    cv::imshow("Display Mosaic Reconstruction", mosaic);
    cv::waitKey(0);
}

void MosaicCamera::Panorama(cv::Mat refRotation, std::vector<Frame *>frames) {
    
    cv::detail::MultiBandBlender blender;

    blender.prepare(cv::Rect{0, 0, 1000, 1000});

    cv::Mat mask, bigImage, curImage;

    std::vector<cv::Point2f> corners(4);

    for (Frame *f : frames) {
        cv::Mat R = refRotation * f->R;
        cv::warpPerspective(f->img, curImage, f->K * R * f->K.inv(),
                    cv::Size(1000,1000), cv::INTER_LINEAR, cv::BORDER_TRANSPARENT);
        corners[0] = cv::Point2f(0,0);
        corners[1] = cv::Point2f(0,f->img.size().height);
        corners[2] = cv::Point2f(f->img.size().width, 0);
        corners[3] = cv::Point2f(f->img.size().width, f->img.size().height);
        mask = makeMask(curImage.size(), corners, f->K * R * f->K.inv());
        blender.feed(curImage.clone(), mask, cv::Point(0, 0));
    }

    blender.blend(bigImage, mask);
    bigImage.convertTo(bigImage, (bigImage.type() / 8) * 8);
    cv::imshow("Result", bigImage);
    cv::waitKey(0);
}

cv::Mat MosaicCamera::makeMask(cv::Size sz, std::vector<cv::Point2f> imageCorners, cv::Mat H) {
    using namespace cv;
    Scalar white(255, 255, 255);
    Mat mask = Mat::zeros(sz, CV_8U);
    Point2f innerPoint;
    std::vector<Point2f> transformedCorners(4);

    perspectiveTransform(imageCorners, transformedCorners, H);
    // Calculate inner point
    for (auto& point : transformedCorners)
        innerPoint += point;
    innerPoint.x /= 4;
    innerPoint.y /= 4;

    // Make indent for each corner
    std::vector<Point> corners;
    // for (int ind = 0; ind < 4; ++ind) {
    //     Point2f direction = innerPoint - transformedCorners[ind];
    //     double normOfDirection = norm(direction);
    //     corners[ind].x += settings.indent * direction.x / normOfDirection;
    //     corners[ind].y += settings.indent * direction.y / normOfDirection;
    // }

    // Draw borders
    Point prevPoint = corners[3];
    for (auto& point : corners) {
        line(mask, prevPoint, point, white);
        prevPoint = point;
    }

    // Fill with white
    floodFill(mask, innerPoint, white);
    return mask;
}