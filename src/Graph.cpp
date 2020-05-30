#include "Graph.h"
#include "Initializer.h"

void Graph::AddFrame(cv::Mat img) {
    Frame *frame = new Frame(frames.size(), (frames.size() == 0), img, (frames.size() == 0) ? nullptr : frames.back());
    Initializer initializer(frame);
    initializer.initialize(points);
    std::cout << "loading frame: " << frames.size() << std::endl;
    frame->printStatus();
    if (frame->isFirst) {
        frames.push_back(frame);
        std::cout << "add frame (" << frame->frameId << ") to the graph" << std::endl;
    }
    if (frame->inliers.size() > 100){
        frames.push_back(frame);
        std::cout << "add frame (" << frame->frameId << ") to the graph" << std::endl;
    }
}
/*
we want to use ceres::Solve function to optimize our graph bundle adjustment

To do so we need to define three components:
- ceres::Summary summary, variable that stores the report of our bundle adjustment
- ceres::Option options, variable that stores the configuration of our optimization
- ceres::Problem problem, variable that defines the graph structure and cost function
*/
void Graph::Optimize() {

    ceres::Problem problem;
    for (std::pair<int, Point *> p : points) {
        for (std::pair<Frame *, int> obs: p.second->observations) {
            // convert this to angle-axis
            cv::Mat pts = cv::Mat(p.second->pt);
            
            // Each Residual block takes a point and a camera as input and outputs a 2
            // dimensional residual. Internally, the cost function stores the observed
            // image location and compares the reprojection against the observation.
            Frame *frame = obs.first;
            double var1 = static_cast<double>(obs.first->kpts[obs.second].pt.x);
            double var2 = static_cast<double>(obs.first->kpts[obs.second].pt.y);

            double K[4];
            K[0] = static_cast<double>(Frame::K.at<float>(0,0));
            K[1] = static_cast<double>(Frame::K.at<float>(1,1));
            K[2] = static_cast<double>(Frame::K.at<float>(0,2));
            K[3] = static_cast<double>(Frame::K.at<float>(1,2));

            float residuals[2];
            ceres::CostFunction* cost_function =
                SphericalReprojectionError::Create(var1,  // index of the camera
                                                   var2,
                                                   K); // index of the point
            double camera[3], scenePts[3];
            camera[0] = static_cast<double>(frame->angleAxis[0]);
            camera[1] = static_cast<double>(frame->angleAxis[1]);
            camera[2] = static_cast<double>(frame->angleAxis[2]);

            scenePts[0] = static_cast<double>(p.second->X[0]);
            scenePts[1] = static_cast<double>(p.second->X[1]);
            scenePts[2] = static_cast<double>(p.second->X[2]);

            problem.AddResidualBlock(cost_function, 
                                    NULL /* squared loss */,
                                    camera, // float[3]
                                    scenePts);
        }
    }

    // we also want to adjust our initial estimation of rotation matrix and 3d spherical scene points during this bundle adjustment
    // Make Ceres automatically detect the bundle structure. Note that the
    // standard solver, SPARSE_NORMAL_CHOLESKY, also works fine but it is slower
    // for standard bundle adjustment problems.
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";

}

Graph::Graph() { }

Graph::~Graph() {
    for (auto p : points) delete p.second;
    for (auto f : frames) delete f;
}
