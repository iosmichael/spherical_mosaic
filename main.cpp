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

using namespace cv;

void readImages(char *dirpath, std::vector<std::string> &image_path);

void test();
void testGraph(std::vector<std::string> &img_path);

float fx = 2569.31, fy = 2566.96;
float cx = 2755.64, cy = 1788.42;
cv::Mat Frame::K = (cv::Mat_<float>(3,3) << fx,0,cx,0,fy,cy,0,0,1);


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
    /*
    std::vector<Frame *> frames;    
    
    int countId = 0;
    Frame *prevFrame = nullptr;
    for (auto path : img_path) {
        cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
        Frame *newFrame = new Frame(countId, (countId == 0), img, prevFrame);
        Initializer initializer(newFrame);
        initializer.initialize();
        frames.push_back(newFrame);
        prevFrame = newFrame;
        std::cout << "loading frame: " << countId << std::endl;
        newFrame->printStatus();
        if (countId > 9) {
            break;
        }
        countId ++;
    }

    // run algorithm here

    // Graph Creation

    std::cout << "total frame loaded in the memory: " << frames.size() << std::endl;
    // release the memory
    for (auto framePtr : frames) {
        delete framePtr;
    }
    */
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
    Frame *prevFrame = nullptr;
    for (auto path : img_path) {
        cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
        graph.AddFrame(img);
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