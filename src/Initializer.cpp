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

void Initializer::RANSAC() {
     float cMinimalCost = 1e10;
     int maxTrials = 100;
     float threshold, tolerance, cost;
     cv::Mat optSol;
     std::vector<cv::DMatch> inliers;
     for (int trials = 0; trials < maxTrials & cMinimalCost > threshold; trials++) {
        // select a random sample from the matches
        int random = std::rand() % frame->matches.size(); 
        cv::DMatch match1 = frame->matches[random];
        random = std::rand() % frame->matches.size(); 
        cv::DMatch match2 = frame->matches[random];
        cv::Mat minimumSol;
        
        MinimumSolverCalibratedRotation(match1, match2, minimumSol);
        cost = ComputeCost();
        // use the matching feature coordinates to calculate model estimation: rotation
        // use the rotation to calculate all the matches in term of reprojection error
        if (cost < cMinimalCost) {
            cMinimalCost = cost;
            minimumSol.copyTo(optSol);
        }
        // save the optimal solution
     }
     // calculate the final error of optimal model
     // calculate the inliers using the optimal model
}

float Initializer::ComputeCost() {
    /*
    Perform reprojection error, we want to estimate the homography based on our rotation matrix and calibration matrix
    Since we assume that there is no camera translation between frames, we get:

    x' = K * R * K^-1 x
    cost = ||x' - x||_2^2, where x and x' is all the feature matches between reference frame and current frame
    */
}

void Initializer::MinimumSolverCalibratedRotation(cv::DMatch &match1, cv::DMatch &match2, cv::Mat &solution) {
    /*
    B = [x1^T, x2^T, ..., xn^T], where n = 2, B: 2x3 matrix, points in B come from ref frame
    C = [x1, x2, ..., xn], where n = 2, C: 3x2 matrix, points in C come from current frame

    1) S = C.T * B, S: 3x3 matrix
    2) perform singular value decomposition on S matrix, where S = U Sigma V.T
    3) if det(U) * det(V) < 0: R = U * diag(1, 1, -1) * V.T
    4) if det(U) * det(V) > 0: R = U * V.T
    5) assert that det(R) == 1
    */
}

void Initializer::initRotation() {
    /*
    B = [x1^T, x2^T, ..., xn^T], where n = numOfMatches, B: nx3 matrix, points in B come from ref frame
    C = [x1, x2, ..., xn], where n = numOfMatches, C: 3xn matrix, points in C come from current frame

    1) S = C.T * B, S: 3x3 matrix
    2) perform singular value decomposition on S matrix, where S = U Sigma V.T
    3) if det(U) * det(V) < 0: R = U * diag(1, 1, -1) * V.T
    4) if det(U) * det(V) > 0: R = U * V.T
    5) assert that det(R) == 1
    */
}
