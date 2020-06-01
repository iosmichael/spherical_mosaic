#include "Frame.h"

Frame::Frame(int frameId, bool isFirst, cv::Mat &imgData, Frame *refFrame):frameId(frameId), isFirst(isFirst), hasKeyPoints(false), isInitialize(false), refFrame(refFrame) {
    if (isFirst){
        R = cv::Mat::eye(3,3,CV_32F);
        angleAxis[0] = angleAxis[1] = angleAxis[2] = 0;
        refR = cv::Mat::eye(3,3,CV_32F);
    }
    imgData.copyTo(img);
}

Frame::~Frame() { delete angleAxis; }

void Frame::visualize() {
    assert(isInitialize);
    cv::Mat refFrameImg = refFrame->img.clone();

    for( int i = 0; i < (int)matches.size(); i++ )
    {
        printf( "-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, matches[i].queryIdx,  matches[i].trainIdx ); 

        //query image is the first frame
        cv::Point2f point_old = refFrame->kpts[matches[i].trainIdx].pt;
        //train  image is the next frame that we want to find matched keypoints
        cv::Point2f point_new = kpts[matches[i].queryIdx].pt;

        //keypoint color for frame 1: RED
        cv::circle(refFrameImg, point_old, 3, cv::Scalar(0, 0, 255), 0.5);  
        // cv::circle(frameImg, point_old, 3, cv::Scalar(0, 0, 255), 1);  
        
        //keypoint color for frame 2: BLUE
        cv::circle(refFrameImg, point_new, 3, cv::Scalar(255, 0, 0), 0.5);  
        // cv::circle(frameImg, point_new, 3, cv::Scalar(255, 0, 0), 1); 

        //draw a line between keypoints
        cv::line(refFrameImg, point_old, point_new, cv::Scalar(0, 255, 0), 2, 8, 0);
        // cv::line(frameImg, point_old, point_new, cv::Scalar(0, 255, 0), 2, 8, 0); 
    }
    cv::imshow("Reference Frame", refFrameImg);
    cv::waitKey();

    refFrameImg = refFrame->img.clone();
    for( int i = 0; i < (int)inliers.size(); i++ )
    {
        int refIndex = std::get<0>(inliers[i]), currIndex = std::get<1>(inliers[i]);

        //query image is the first frame
        cv::Point2f point_old = refFrame->kpts[refIndex].pt;
        //train  image is the next frame that we want to find matched keypoints
        cv::Point2f point_new = kpts[currIndex].pt;

        //keypoint color for frame 1: RED
        cv::circle(refFrameImg, point_old, 3, cv::Scalar(0, 0, 255), 0.5);  
        // cv::circle(frameImg, point_old, 3, cv::Scalar(0, 0, 255), 1);  
        
        //keypoint color for frame 2: BLUE
        cv::circle(refFrameImg, point_new, 3, cv::Scalar(255, 0, 0), 0.5);  
        // cv::circle(frameImg, point_new, 3, cv::Scalar(255, 0, 0), 1); 

        //draw a line between keypoints
        cv::line(refFrameImg, point_old, point_new, cv::Scalar(0, 255, 0), 2, 8, 0);
        // cv::line(frameImg, point_old, point_new, cv::Scalar(0, 255, 0), 2, 8, 0); 
    }
    cv::imshow("Reference Frame After Outlier Rejection", refFrameImg);
    cv::waitKey();
}

void Frame::printStatus() {
    std::cout << "Frame status: ID " << frameId << std::endl;
    std::cout << "\tisFirst: " << isFirst << std::endl;
    std::cout << "\tisInitialize: " << isInitialize << std::endl;
    std::cout << "\tnumOfKeyPoints: " << kpts.size() << std::endl;
    std::cout << "\tnumOfMatchesToReferenceFrame: " << matches.size() << std::endl;
    std::cout << "\timageSize: (w: " << img.cols << ", h:" << img.rows << ")" << std::endl;
}
