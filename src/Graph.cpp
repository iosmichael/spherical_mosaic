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

Graph::~Graph() {
    for (auto p : points) delete p;
    for (auto f : frames) delete f;
}
