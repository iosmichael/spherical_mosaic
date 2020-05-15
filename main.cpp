/*
main program of spherical mosaic final project

author: Michael, Suhrid
date: 5/13/2020
*/

#include <iostream>
#include <string>
#include <boost/filesystem.hpp>
#include <vector>
#include <opencv/cv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;

void featureDetections(Mat &src, Mat &tar, std::vector<KeyPoint> &kpt1, std::vector<KeyPoint> &kpt2, Mat &desc1, Mat &desc2);
void featureMatching(std::string type, Mat &desc1, Mat &desc2, std::vector<DMatch>& matches);
void drawFeatureCorrespondence(Mat &src, Mat &tar, std::vector<DMatch> &good_matches, std::vector<KeyPoint> &kpt1, std::vector<KeyPoint> &kpt2);

void printInstruction();
void readImages(char *dirpath, std::vector<std::string> &image_path);

int main(int argc, char** argv) {

    Mat gray1, gray2;
    if (argc != 2)
        return -1;
    std::cout << argv[0] << std::endl;
    std::cout << argv[1] << std::endl;

    std::vector<std::string> img_path;
    
    readImages(argv[1], img_path);
    std::sort(img_path.begin(), img_path.end());
    
    for (auto path : img_path) {
        std::cout << path << std::endl;
    }
    
    std::string img1 = img_path.at(0);
    std::string img2 = img_path.at(1);
    gray1 = cv::imread(img1, cv::IMREAD_GRAYSCALE);
    gray2 = cv::imread(img2, cv::IMREAD_GRAYSCALE);

    std::vector<KeyPoint> kpt1, kpt2;
    cv::Mat desc1, desc2;
    std::vector<cv::DMatch> matches;
    featureDetections(gray1, gray2, kpt1, kpt2, desc1, desc2);
    featureMatching("knn", desc1, desc2, matches);
    drawFeatureCorrespondence(gray1, gray2, matches, kpt1, kpt2);

    
    return 0;
}

void featureDetections(Mat &src, Mat &tar, std::vector<KeyPoint> &kpt1, std::vector<KeyPoint> &kpt2, Mat &desc1, Mat &desc2){
    cv::Ptr<cv::xfeatures2d::SIFT> detector = cv::xfeatures2d::SIFT::create();
    detector -> detectAndCompute(src, Mat(), kpt1, desc1);
    detector -> detectAndCompute(tar, Mat(), kpt2, desc2);   
    Mat img_keypoints;
    drawKeypoints(src, kpt1, img_keypoints);
    //-- Show detected (drawn) keypoints
    imshow("SIFT Keypoints", img_keypoints);
    waitKey();
}

void featureMatching(std::string type, Mat &desc1, Mat &desc2, std::vector<DMatch>& matches){
    const double kDistanceCoef = 4.0;
    const int kMaxMatchingSize = 50;
    
    matches.clear();
    if (type == "bf")
    {
        BFMatcher des_matcher(cv::NORM_L2, true);
        des_matcher.match(desc1, desc2, matches, Mat());
    }
    if (type == "knn") {
        BFMatcher desc_matcher(cv::NORM_L2, true);
        std::vector<std::vector<DMatch>> vmatches;
        desc_matcher.knnMatch(desc1, desc2, vmatches, 1);
        for (int i = 0; i < static_cast<int>(vmatches.size()); ++i) {
            if (!vmatches[i].size()) {
                continue;
            }
            matches.push_back(vmatches[i][0]);
        }
    }
    std::sort(matches.begin(), matches.end());
    while (matches.front().distance * kDistanceCoef < matches.back().distance) {
        matches.pop_back();
    }
    while (matches.size() > kMaxMatchingSize) {
        matches.pop_back();
    }
}

void drawFeatureCorrespondence(Mat &src, Mat &tar, std::vector<DMatch> &good_matches, std::vector<KeyPoint> &kpt1, std::vector<KeyPoint> &kpt2){
    for( int i = 0; i < (int)good_matches.size(); i++ )
    {
    printf( "-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, good_matches[i].queryIdx,  good_matches[i].trainIdx ); 

    //query image is the first frame
    Point2f point_old = kpt1[good_matches[i].queryIdx].pt;

    //train  image is the next frame that we want to find matched keypoints
    Point2f point_new = kpt2[good_matches[i].trainIdx].pt;

    //keypoint color for frame 1: RED
    circle(src, point_old, 3, Scalar(0, 0, 255), 1);  
    circle(tar, point_old, 3, Scalar(0, 0, 255), 1);  

    //keypoint color for frame 2: BLUE
    circle(src, point_new, 3, Scalar(255, 0, 0), 1);  
    circle(tar, point_new, 3, Scalar(255, 0, 0), 1); 

    //draw a line between keypoints
    line(src, point_old, point_new, Scalar(0, 255, 0), 2, 8, 0);
    line(tar, point_old, point_new, Scalar(0, 255, 0), 2, 8, 0); 
    }
    imshow("Image 1", src);
    waitKey();
}

void printInstruction(){
    // std::cout << "[FLAG] img_directory" << std::endl;
}

void readImages(char *dirpath, std::vector<std::string> &image_path) {
    for (boost::filesystem::directory_entry &item : boost::filesystem::directory_iterator(dirpath)){
        std::string ext = boost::filesystem::extension(item.path());
        if (ext == ".png") {
            image_path.push_back(item.path().string());
        }
        if (ext == ".jpg") {
            image_path.push_back(item.path().string());
        }
    }
}