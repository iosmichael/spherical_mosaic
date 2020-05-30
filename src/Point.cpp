#include "Point.h"
#include "Utility.h"

Point::Point(int id): pointID(id) { }

Point::~Point() { delete X; } 
    
void Point::AddObservation(Frame *frame, int index) {
    observations[frame] = index;
    Update();
}
    
void Point::Update() {
    cv::Mat avgPt = cv::Mat::zeros(3,1,CV_32F);
    int n = 0;
    for (auto k : observations) {
        Frame *f = k.first; int kptIndex = k.second;
        cv::Mat p = (cv::Mat_<float>(3,1,CV_32F) << f->kpts[kptIndex].pt.x, f->kpts[kptIndex].pt.y, 1);
        p = f->R * Utility::Normalize(p, f->K);
        avgPt += p;
        n += 1;
    }
    avgPt /= n;
    int sign = (avgPt.at<float>(2,0) < 0) ? -1 : 1;
    avgPt /= cv::norm(avgPt, cv::NORM_L2) * sign;
    pt = cv::Point3f(avgPt);

    // update parameters used for bundle adjustment
    X[0] = static_cast<double>(pt.x);
    X[1] = static_cast<double>(pt.y);
    X[2] = static_cast<double>(pt.z);
}