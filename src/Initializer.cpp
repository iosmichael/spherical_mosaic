#include "Initializer.h"
#include "Utility.h"
#include <ceres/rotation.h>

cv::Ptr<cv::xfeatures2d::SIFT> Initializer::detector = cv::xfeatures2d::SIFT::create();

Initializer::Initializer(Frame *frame): frame(frame) { }

Initializer::~Initializer() { }

void Initializer::initialize(std::map<int, Point *> &scenePoints) {
    if (!frame->isFirst)
    {            
        FeatureExtractor();
        FeatureMatcher();
        SolveCalibratedRotationDLT();
        // frame has refR initialized with a set of inliers
        frame->isInitialize = true;

        // visualize();

        // if frame inliers are related to unseen scene points, initialize the scenepoints
        // if frame inliers are related to seen scene points, update the scenepoints with new observation
        InitializeScenePoints(scenePoints);
    }
}

void Initializer::visualize() {
    // we have frame->refR, the estimated rotation from current frame to reference frame
    cv::Mat display = frame->refFrame->img.clone();
    std::cout << "number of inliers found: " << frame->inliers.size() << std::endl;
    for( int i = 0; i < (int)frame->inliers.size(); i++ )
    {
        int refIndex = std::get<0>(frame->inliers[i]), currIndex = std::get<1>(frame->inliers[i]);
        // printf( "-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, refIndex,  currIndex); 

        //query image is the first frame
        cv::Point2f point_old = frame->refFrame->kpts[refIndex].pt;
        //train  image is the next frame that we want to find matched keypoints
        cv::Mat point_curr = (cv::Mat_<float>(3,1,CV_32F) << frame->kpts[currIndex].pt.x, frame->kpts[currIndex].pt.y, 1);
        point_curr = Utility::Dehomogenize(frame->K * frame->refR * Utility::Normalize(point_curr, frame->K));
        cv::Point2f point_new = cv::Point2f(point_curr);

        //keypoint color for frame 1: RED
        cv::circle(display, point_old, 3, cv::Scalar(0, 0, 255), 1);  
        
        //keypoint color for frame 2: BLUE
        cv::circle(display, point_new, 3, cv::Scalar(255, 0, 0), 1);  

        //draw a line between keypoints
        cv::line(display, point_old, point_new, cv::Scalar(0, 255, 0), 2, 8, 0);
    }
    cv::imshow("Visualize Reprojection Frame", display);
    cv::waitKey();
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
    // here our train is the features from the reference frame
    // our query is the features from the current frame
    descMatcher.knnMatch(frame->desc, frame->refFrame->desc, vmatches, 1);
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

    RANSAC();
}

void Initializer::RANSAC() {
     float cMinimalCost = 1e10;
     int maxTrials = 100;
     // tolerance = 5.991464547107979
     float threshold = 100, tolerance = 5.991464547107979, cost=0;
     float p = 0.99, alpha = 0.95, w = 0;

     int minMatches = 2;
     cv::Mat optSol;
     std::vector<std::tuple<int, int>> tempInliers;
     for (size_t trials = 0; trials < maxTrials & cMinimalCost > threshold; trials++) {
        // select a random sample from the matches
        std::vector<cv::DMatch> selected_matches;
        for (size_t i = 0; i < minMatches; i++) {
            std::random_shuffle(frame->matches.begin(), frame->matches.end());
            selected_matches.push_back(frame->matches[0]);
            selected_matches.push_back(frame->matches[1]);
        }
        
        cv::Mat minimumSol;
        
        SolveCalibratedRotation(selected_matches, minimumSol);

        cost = ComputeCost(minimumSol, tolerance, tempInliers);

        // std::cout << cost << std::endl;

        // use the matching feature coordinates to calculate model estimation: rotation
        // use the rotation to calculate all the matches in term of reprojection error
        if (cost < cMinimalCost) {
            cMinimalCost = cost;
            minimumSol.copyTo(optSol);
            // maxTrials = std::log(1-p) / std::log(1-std::)
        }
        // adaptive values
     }
     // calculate the final error of optimal model
     // calculate the inliers using the optimal model
     cost = ComputeCost(optSol, tolerance, frame->inliers);
     std::cout << "final cost for RANSAC rotation: " << cost << std::endl;
}

float Initializer::ComputeCost(cv::Mat &R, float tolerance, std::vector<std::tuple<int, int>> &inliers) {
    /*
    Perform reprojection error, we want to estimate the homography based on our rotation matrix and calibration matrix
    Since we assume that there is no camera translation between frames, we get:

    x' = K * R * K^-1 x
    cost = ||x' - x||_2^2, where x and x' is all the feature matches between reference frame and current frame
    */

    cv::Mat refPts, currPts;
    for (auto m : frame->matches) {
        cv::Point2f refPt = frame->refFrame->kpts[m.trainIdx].pt, curPt = frame->kpts[m.queryIdx].pt;
        cv::Mat x = (cv::Mat_<float>(1,3) << refPt.x, refPt.y, 1), xp = (cv::Mat_<float>(1,3) << curPt.x, curPt.y, 1);
        refPts.push_back(x);
        currPts.push_back(xp);
    }

    cv::Mat xp = frame->K * R * Utility::Normalize(currPts.t(), frame->K); // x is homogenous
    xp = Utility::Dehomogenize(xp); refPts = Utility::Dehomogenize(refPts.t()); 
    
    float cost = 0;
    inliers.clear();
    for (size_t i = 0; i < xp.cols; i++) {
        double norm = cv::norm(xp.col(i) - refPts.col(i), cv::NORM_L2);
        if (norm > tolerance) {
            cost += tolerance;
        } else {
            cost += norm;
            inliers.push_back(std::tuple<int, int>(frame->matches[i].trainIdx, frame->matches[i].queryIdx));
        }
    }
    return cost;
}

void Initializer::SolveCalibratedRotation(std::vector<cv::DMatch> matches, cv::Mat &solution) {
    /*
    B = [x1^T, x2^T, ..., xn^T], where n = 2, B: 2x3 matrix, points in B come from ref frame
    C = [x1, x2, ..., xn], where n = 2, C: 3x2 matrix, points in C come from current frame

    1) S = C.T * B, S: 3x3 matrix
    2) perform singular value decomposition on S matrix, where S = U Sigma V.T
    3) if det(U) * det(V) < 0: R = U * diag(1, 1, -1) * V.T
    4) if det(U) * det(V) > 0: R = U * V.T
    5) assert that det(R) == 1
    */

    // estimating the rotation from current to previous frame
    cv::Mat B, C;
    for (auto m : matches) {
        // need to double check whether the query and train associate with the reference frame and current frame
        cv::KeyPoint kptRef = frame->refFrame->kpts[m.trainIdx], kptCurr = frame->kpts[m.queryIdx];
        cv::Mat x = (cv::Mat_<float>(1,3) << kptRef.pt.x, kptRef.pt.y, 1);
        cv::Mat xp = (cv::Mat_<float>(1,3) << kptCurr.pt.x, kptCurr.pt.y, 1);
        B.push_back(xp); // n x 3
        C.push_back(x); // n x 3
    }

    B = B.t(); C = C.t(); // 3 x n
    B = Utility::Normalize(B, frame->K); C = Utility::Normalize(C, frame->K);
    cv::Mat S = C * B.t();
    
    cv::SVD s;
    cv::Mat u, sigma, vt;
    s.compute(S, sigma, u, vt);
    if ((cv::determinant(u) * cv::determinant(vt)) < 0){
        cv::Mat diag = cv::Mat::eye(3, 3, CV_32F);
        diag.at<float>(2,2) = -1;
        solution = u * diag * vt;
    } else {
        solution = u * vt;
    }
    // std::cout << "R determinant: " << cv::determinant(solution) << std::endl;
}

void Initializer::SolveCalibratedRotationDLT() {
    // B is the homogeneous coordinate of points in the current frame
    // C is the homogeneous coordinate of points in the reference frame
    cv::Mat B, C;
    for (auto t: frame->inliers) {
        cv::Mat x = (cv::Mat_<float>(1,3) << frame->refFrame->kpts[std::get<0>(t)].pt.x, frame->refFrame->kpts[std::get<0>(t)].pt.y, 1);
        cv::Mat xp = (cv::Mat_<float>(1,3) << frame->kpts[std::get<1>(t)].pt.x, frame->kpts[std::get<1>(t)].pt.y, 1);
        B.push_back(xp); // n x 3
        C.push_back(x); // n x 3
    }
    
    B = B.t(); C = C.t(); // 3 x n
    B = Utility::Normalize(B, frame->K); C = Utility::Normalize(C, frame->K);
    cv::Mat S = C * B.t();

    cv::SVD s;
    cv::Mat u, sigma, vt;
    s.compute(S, sigma, u, vt);
    if ((cv::determinant(u) * cv::determinant(vt)) < 0){
        cv::Mat diag = cv::Mat::eye(3, 3, CV_32F);
        diag.at<float>(2,2) = -1;
        frame->refR = u * diag * vt;
    } else {
        frame->refR = u * vt;
    }
    frame->R = frame->refFrame->R * frame->refR;
    // std::cout << "R determinant: " << cv::determinant(frame->refR) << std::endl;
    double R[9] = {0};
    for(size_t i = 0; i < 3; ++i)
    {
        for(size_t j = 0; j < 3; ++j)
        {
            R[i * 3 + j] = frame->R.at<float>(i, j);
        }
    }
    ceres::RotationMatrixToAngleAxis(R, frame->angleAxis);
}

void Initializer::InitializeScenePoints(std::map<int, Point *> &scenePoints) {
    // <int, point pointer> : int is our kpt index
    // <int, int> : t, where the first element is reference kpts index, the second element is current kpts index
    int createdCount = 0, updatedCount = 0;
    for (auto t: frame->inliers) {
        if (frame->refFrame->scenePts.count(std::get<0>(t))) {
            // reference frame has already initialized the scenePts
            int sP = frame->refFrame->scenePts[std::get<0>(t)];
            frame->scenePts[std::get<1>(t)] = sP;
            scenePoints[sP]->AddObservation(frame, (int) std::get<1>(t));
            updatedCount += 1;
        } else {
            int pID = frame->frameId + scenePoints.size() + 1e3;
            Point *sP = new Point(pID);
            frame->scenePts[std::get<1>(t)] = pID;
            frame->refFrame->scenePts[std::get<0>(t)] = pID;
            sP->AddObservation(frame->refFrame, (int) std::get<0>(t));
            sP->AddObservation(frame, (int) std::get<1>(t));
            scenePoints[pID] = sP;
            createdCount += 1;
        }
    }
    std::cout << createdCount << " of points created" << std::endl;
    std::cout << updatedCount << " of points updated" << std::endl;
}