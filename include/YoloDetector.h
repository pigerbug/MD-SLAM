#ifndef YOLODETECTOR_H
#define YOLODETECTOR_H

//#include "Tracking.h"
// #include "benchmark.h"
// #include "cpu.h"
// #include "datareader.h"
// #include "net.h"
//#include <torch/script.h>
#include <utility>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <stdio.h>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <unistd.h>
#include <mutex>
#include <chrono>
#include <vector>
#include <set>
#include "NvInferRuntime.h"

#include <cv.h>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/line_descriptor/descriptor.hpp>
using namespace nvinfer1;
using namespace std;
namespace ORB_SLAM2
{
class Tracking;

class YoloBoundingBox{
private:
    cv::Rect2f rect;
    std::string label;
    //This id was used for multi object tracking, but in experiment I found that MOT cost too much time, so I delete the MOT module
    //Right now, this id only used for draw colors of bbox
    int id;
    float score;

public:
    YoloBoundingBox(cv::Rect2f input_rect, std::string input_label, float score);
    YoloBoundingBox(float x1, float y1, float x2, float y2, std::string input_label, float score);
    const std::string& GetLabel() const {return this->label;}
    cv::Rect2f GetRect() const {return this->rect;}
    float GetScore() const {return this->score;}
    int GetId() const {return this->id;}
    void SetId(int inputId){this->id = inputId;}

    float average, median, stdcov, average_person;
    float width, height;
    std::vector<float> mvKeysDynam;
    std::vector<float> mvKeysDynamIntensity;
    
    std::vector<pair<int,cv::KeyPoint>> KeyPointsInBox;
    cv::Mat CorrespongdingDiscriptors;
    std::vector<double> epiErr,epiErrChi2;
    bool isDynamic = true;
    double proStatic = 0.0;
    //line
    std::vector<pair<int,cv::line_descriptor::KeyLine>> KeyLinesInBox;
    std::vector<double> epiErr_line,epiErrChi2_line;
    std::vector<pair<float,float>> mvLinesDynam;
    std::vector<float> mvLinesDynamIntensity;
    //depth
    std::vector<float> mvdep;
    float dep_average, dep_stdcov;
    float dep_sum;
    pair<bool, bool> box_semantic;

    std::vector<double> epiErr_box;
    std::vector<cv::Point2f> key_in_box;

    float key_dep_average, key_dep_stdcov;
};

class YoloDetector
{
private:
    int cpuThreadNum;
    // ncnn::Net detector;
    
    std::vector<std::string> class_names;
public:
    YoloDetector();
    void DetectByTensorRT(cv::Mat& image, cv::Mat& depth, std::vector<YoloBoundingBox>& yoloBoundingBoxList);

    void SetTracker(Tracking* pTracker);

    bool isNewImgArrived();
    void Run();
    bool CheckFinish();
    void RequestFinish();

    void doInference(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* output, int batchSize);

    std::mutex mMutexNewYoloDetector;
    std::mutex mMutexGetNewImg;
    std::mutex mMutexFinish;
    bool mbNewImgFlag;
    bool mbFinishRequested;
    Tracking* mpTracker;
    cv::Mat mImg;
    cv::Mat mDepth;
    bool mbTensorRT;
    bool mbYOLO;
    
//    torch::jit::script::Module mModule;

    IExecutionContext* context;
};

}

#endif
