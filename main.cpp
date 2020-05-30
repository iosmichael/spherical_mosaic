/*
main program of spherical mosaic final project

author: Michael, Suhrid
date: 5/13/2020
*/

#include <iostream>
#include <string>
#include <boost/filesystem.hpp>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv/cv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/highgui.hpp>

#include "src/Frame.h"
#include "src/Graph.h"
#include "src/Initializer.h"
#include "src/Mosaic.h"
#include "src/Utility.h"

using namespace cv;

void readImages(char *dirpath, std::vector<std::string> &image_path);

void test();
void testGraph(std::vector<std::string> &img_path);
void testSphericalMapping(std::vector<std::string> &img_path);

size_t downScale = 5;

float fx = 2566.96, fy = 2569.31;
float cx = 1788.42, cy = 2755.64;
cv::Mat Frame::K = (cv::Mat_<float>(3,3) << fx / downScale,0,cx / downScale,0,fy / downScale,cy / downScale,0,0,1);


int main(int argc, char** argv) {
    // import calibration matrix K from dataset lunchroom
    std::cout << "Calibration Matrix: " << Frame::K << std::endl;
    // test();

    if (argc != 2)
        return -1;

    std::vector<std::string> img_path;
    
    readImages(argv[1], img_path);
    std::sort(img_path.begin(), img_path.end());
    testGraph(img_path);

    // testSphericalMapping(img_path);
    return 0;
}

void test() {
    cv::Mat B;
    cv::Point2f x = cv::Point2f{2,3}, xp = cv::Point2f{3,4};
    cv::Mat pt = (cv::Mat_<float>(1,3) << x.x, x.y, 1);
    std::cout << pt << std::endl;
    B.push_back(pt);
    B.push_back(pt);
    std::cout << B << std::endl;
    B = B.t();
    std::cout << B << std::endl;
    for (size_t i = 0; i < B.cols; i++) {
        B.at<float>(0, i) = B.at<float>(0, i) / B.at<float>(2, i);
        B.at<float>(1, i) = B.at<float>(1, i) / B.at<float>(2, i);
    }
    B = B.rowRange(cv::Range(0,2));
    std::cout << B << std::endl;
}

void testGraph(std::vector<std::string> &img_path) {
    Graph graph = Graph();
    
    int countId = 0;

    for (auto path : img_path) {
        cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
        cv::resize(img, img, cv::Size(img.cols / downScale, img.rows / downScale));
        cv::Mat undistortImg, mappedImg;
        
        //undistort the image
        cv::Mat coeffs = (cv::Mat_<float>(1,4,CV_32F) << -0.0484573, 0.0100024, 0.00050623, 0.000603611);
        cv::InputArray distortCoeffs = cv::InputArray(coeffs);
        cv::undistort(img, undistortImg, Frame::K, distortCoeffs);
        
        // cv::imshow("display original img", img);
        // cv::waitKey(0);
        // cv::imshow("display undistorted img", undistortImg);
        // cv::waitKey(0);

        graph.AddFrame(undistortImg);
        countId += 1;
        if (countId >= 15) break;
    }

    for (auto f : graph.frames) {
        // f->visualize();
        std::cout << "rotation of frame " << f->frameId << ": " << f->R << std::endl;
    }

    graph.Optimize();
    // MosaicCamera mosaic {90,100,100,cv::Mat::eye(cv::Size(3,3), CV_32F)};
    // mosaic.Visualize(cv::Mat::eye(cv::Size(3,3), CV_32F), graph.frames);
}

void testSphericalMapping(std::vector<std::string> &img_path) {
    Graph graph = Graph();
    
    for (auto imgpath: img_path){
        cv::Mat img = cv::imread(imgpath, cv::IMREAD_COLOR);
        cv::resize(img, img, cv::Size(img.cols / 5, img.rows / 5));
        cv::Mat undistortImg, mappedImg;
        cv::imshow("display original img", img);
        cv::waitKey(0);
        
        //undistort the image
        cv::Mat coeffs = (cv::Mat_<float>(1,4,CV_32F) << -0.0484573, 0.0100024, 0.00050623, 0.000603611);
        cv::InputArray distortCoeffs = cv::InputArray(coeffs);
        cv::undistort(img, undistortImg, Frame::K, distortCoeffs);
        cv::imshow("display undistorted img", undistortImg);
        cv::waitKey(0);

        cv::Mat xmap = cv::Mat::zeros(cv::Size(undistortImg.cols, undistortImg.rows), CV_32F), ymap = cv::Mat::zeros(cv::Size(undistortImg.cols, undistortImg.rows), CV_32F);

        float kCx = Frame::K.at<float>(0,2), kCy = Frame::K.at<float>(1,2);
        float kFx = Frame::K.at<float>(0,0), kFy = Frame::K.at<float>(1,1);
        for (size_t i = 0; i < undistortImg.rows; i++) {
            for (size_t j = 0; j < undistortImg.cols; j++) {
                 float theta = (j - kCx) / kFx;
                 float phi = (i - kCy) / kFy;
                 cv::Mat X = (cv::Mat_<float>(3,1,CV_32F) << std::sin(theta) * std::cos(phi), std::sin(phi), std::cos(theta) * std::cos(phi));
                 X = Utility::Dehomogenize(Frame::K * X);
                 xmap.at<float>(i,j) = X.at<float>(0,0);
                 ymap.at<float>(i,j) = X.at<float>(1,0);
            }
        }
        cv::Mat remappedImg;
        cv::remap(undistortImg, remappedImg, xmap, ymap, cv::INTER_LINEAR, cv::BORDER_DEFAULT);
        cv::imshow("remapped img", remappedImg);
        cv::waitKey(0);
    }
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