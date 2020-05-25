#include "Graph.h"
#include "Initializer.h"

void Graph::AddFrame(cv::Mat img) {
    Frame *frame = new Frame(frames.size(), (frames.size() == 0), img, (frames.size() == 0) ? nullptr : frames.back());
    frames.push_back(frame);
    Initializer initializer(frame);
    initializer.initialize(points);
    std::cout << "loading frame: " << frames.size() << std::endl;
    frame->printStatus();
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
            cv::Mat Rinv = obs.first->R.inv(); // rotation from the current frame to the first frame
            
            // convert this to angle-axis
            cv::Mat pts = cv::Mat(p.second->pt);
            
            // Each Residual block takes a point and a camera as input and outputs a 2
            // dimensional residual. Internally, the cost function stores the observed
            // image location and compares the reprojection against the observation.
            auto var1 = obs.first->kpts[obs.second].pt.x;
            auto var2 = obs.first->kpts[obs.second].pt.y;

            // ceres::CostFunction* cost_function =
            //     SphericalReprojectionError::Create(observations[2 * i + 0],  // index of the camera
            //                                     observations[2 * i + 1]); // index of the point
            // problem.AddResidualBlock(cost_function, 
            //                         NULL /* squared loss */,
            //                         bal_problem.mutable_camera_for_observation(i), // pointer to the 9 camera parameters
            //                         bal_problem.mutable_point_for_observation(i)); // pointer to the 3 point parameters
        }
    }

    // we also want to adjust our initial estimation of rotation matrix and 3d spherical scene points during this bundle adjustment
}

Graph::Graph() { }

Graph::~Graph() {
    for (auto p : points) delete p.second;
    for (auto f : frames) delete f;
}
