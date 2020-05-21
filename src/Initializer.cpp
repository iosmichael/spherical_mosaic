#include "Initializer.h"

Initializer::Initializer(Frame *frame): frame(frame) { }

Initializer::~Initializer() { }

void Initializer::initialize() {
    if (!frame->isFirst)
    {            
        FeatureExtractor();
        FeatureMatcher();
    }
}

// feature extraction with SIFT detector
// img1, img2
void Initializer::FeatureExtractor() {
    cv::Mat frameGray, refFrameGray;
    cv::cvtColor(frame->img, frameGray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(frame->refFrame->img, refFrameGray, cv::COLOR_BGR2GRAY);

    if (!frame->hasKeyPoints){
        detector -> detectAndCompute(frameGray, cv::Mat(), frame->kpts, frame->desc);
        frame->hasKeyPoints = true;
    }
    if (!frame->refFrame->hasKeyPoints){
        detector -> detectAndCompute(refFrameGray, cv::Mat(), frame->refFrame->kpts, frame->refFrame->desc); 
        frame->refFrame->hasKeyPoints = true;
    }
}

void Initializer::FeatureMatcher() {
    assert(frame->hasKeyPoints && frame->refFrame->hasKeyPoints);
    frame->matches.clear();

    cv::BFMatcher descMatcher(cv::NORM_L2, true);    
    std::vector<std::vector<cv::DMatch>> vmatches;

    // first argument is the feature descriptor from the query image, the second argument is the feature descriptor from the train image. 
    // here our query is the features from the reference frame
    // our train is the features from the current frame
    descMatcher.knnMatch(frame->refFrame->desc, frame->desc, vmatches, 1);
    for (int i = 0; i < static_cast<int>(vmatches.size()); ++i) {
        if (!vmatches[i].size()) {
            continue;
        }
        frame->matches.push_back(vmatches[i][0]);
    }

    std::sort(frame->matches.begin(), frame->matches.end());

    // initial outlier rejection on the matches with a certain window
    while (frame->matches.front().distance * kDistanceCoef < frame->matches.back().distance) {
        frame->matches.pop_back();
    }
    while (frame->matches.size() > kMaxMatchingSize) {
        frame->matches.pop_back();
    }
    // perform initial feature descriptor matching

    // perform outlier rejection on the feature matching

    frame->isInitialize = true;
}

void Initializer::initRotation() {
    
}
