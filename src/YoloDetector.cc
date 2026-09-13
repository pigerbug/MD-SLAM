#include <fstream>
#include "YoloDetector.h"
#include "Tracking.h"
#include <iostream>
#include <stdexcept>
#include <vector>

#define CONF_THRESH 0.5f
#define NMS_THRESH 0.45f
#define ENGINE_PATH "model/yolov8x.engine"

namespace
{
class TrtLogger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* message) noexcept override
    {
        if (severity <= Severity::kWARNING)
            std::cerr << "[TensorRT] " << message << std::endl;
    }
};

TrtLogger gLogger;

size_t tensorVolume(const nvinfer1::Dims& dims)
{
    size_t volume = 1;
    for (int i = 0; i < dims.nbDims; ++i)
        volume *= static_cast<size_t>(dims.d[i]);
    return volume;
}
float intersectionOverUnion(const cv::Rect2f& first, const cv::Rect2f& second)
{
    const float intersection = (first & second).area();
    const float combined = first.area() + second.area() - intersection;
    return combined > 0.0f ? intersection / combined : 0.0f;
}
}

namespace ORB_SLAM2
{

YoloDetector::YoloDetector()
    : cpuThreadNum(1), mbNewImgFlag(false), mbFinishRequested(false), mpTracker(NULL),
      mbTensorRT(true), mbYOLO(false), context(NULL), engine(NULL), stream(NULL),
      buffers{NULL, NULL}, inputIndex(-1), outputIndex(-1), inputHeight(640),
      inputWidth(640), inputSize(0), outputSize(0)
{
    std::ifstream namesFile("model/coco.names");
    std::string name;
    while (std::getline(namesFile, name))
        class_names.push_back(name);

    std::ifstream engineFile(ENGINE_PATH, std::ios::binary);
    if (!engineFile)
        throw std::runtime_error("Unable to open YOLOv8 engine: " ENGINE_PATH);
    engineFile.seekg(0, std::ios::end);
    const size_t engineSize = static_cast<size_t>(engineFile.tellg());
    engineFile.seekg(0, std::ios::beg);
    std::vector<char> serialized(engineSize);
    engineFile.read(serialized.data(), static_cast<std::streamsize>(engineSize));

    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
    if (!runtime)
        throw std::runtime_error("Unable to create TensorRT runtime");
    engine = runtime->deserializeCudaEngine(serialized.data(), engineSize);
    runtime->destroy();
    if (!engine || engine->getNbBindings() != 2)
        throw std::runtime_error("YOLOv8 engine must have one input and one output binding");
    context = engine->createExecutionContext();
    if (!context)
        throw std::runtime_error("Unable to create YOLOv8 execution context");

    for (int binding = 0; binding < engine->getNbBindings(); ++binding)
    {
        if (engine->bindingIsInput(binding))
            inputIndex = binding;
        else
            outputIndex = binding;
    }
    const nvinfer1::Dims inputDims = engine->getBindingDimensions(inputIndex);
    const nvinfer1::Dims outputDims = engine->getBindingDimensions(outputIndex);
    if (inputDims.nbDims != 3 || outputDims.nbDims != 3 || inputDims.d[0] != 3)
        throw std::runtime_error("YOLOv8 engine must use CHW input and 3D output");
    inputHeight = inputDims.d[1];
    inputWidth = inputDims.d[2];
    inputSize = tensorVolume(inputDims);
    outputSize = tensorVolume(outputDims);
    if (outputDims.d[1] < 5 && outputDims.d[2] < 5)
        throw std::runtime_error("Invalid YOLOv8 output shape");

    if (cudaMalloc(&buffers[inputIndex], inputSize * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&buffers[outputIndex], outputSize * sizeof(float)) != cudaSuccess ||
        cudaStreamCreate(&stream) != cudaSuccess)
        throw std::runtime_error("Unable to allocate YOLOv8 CUDA buffers");
    std::cout << "Loaded YOLOv8 TensorRT engine: " << ENGINE_PATH << std::endl;
}

YoloDetector::~YoloDetector()
{
    if (stream)
        cudaStreamDestroy(stream);
    if (buffers[inputIndex])
        cudaFree(buffers[inputIndex]);
    if (buffers[outputIndex])
        cudaFree(buffers[outputIndex]);
    if (context)
        context->destroy();
    if (engine)
        engine->destroy();
}

void YoloDetector::DetectByTensorRT(cv::Mat& image, cv::Mat&, std::vector<YoloBoundingBox>& boxes)
{
    if (image.empty())
        return;

    cv::Mat rgb;
    if (image.channels() == 1)
        cv::cvtColor(image, rgb, cv::COLOR_GRAY2RGB);
    else if (image.channels() == 4)
        cv::cvtColor(image, rgb, cv::COLOR_BGRA2RGB);
    else
        cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);

    const float scale = std::min(inputWidth / static_cast<float>(rgb.cols),
                                 inputHeight / static_cast<float>(rgb.rows));
    const int resizedWidth = static_cast<int>(std::round(rgb.cols * scale));
    const int resizedHeight = static_cast<int>(std::round(rgb.rows * scale));
    const int padX = (inputWidth - resizedWidth) / 2;
    const int padY = (inputHeight - resizedHeight) / 2;
    cv::Mat resized, letterbox(inputHeight, inputWidth, CV_8UC3, cv::Scalar(114, 114, 114));
    cv::resize(rgb, resized, cv::Size(resizedWidth, resizedHeight));
    resized.copyTo(letterbox(cv::Rect(padX, padY, resizedWidth, resizedHeight)));

    std::vector<float> input(inputSize);
    for (int y = 0; y < inputHeight; ++y)
        for (int x = 0; x < inputWidth; ++x)
            for (int channel = 0; channel < 3; ++channel)
                input[channel * inputHeight * inputWidth + y * inputWidth + x] =
                    letterbox.at<cv::Vec3b>(y, x)[channel] / 255.0f;

    std::vector<float> output(outputSize);
    cudaMemcpyAsync(buffers[inputIndex], input.data(), inputSize * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    if (!context->enqueueV2(buffers, stream, NULL))
        throw std::runtime_error("YOLOv8 TensorRT enqueue failed");
    cudaMemcpyAsync(output.data(), buffers[outputIndex], outputSize * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    const nvinfer1::Dims outputDims = engine->getBindingDimensions(outputIndex);
    const int first = outputDims.d[1];
    const int second = outputDims.d[2];
    const bool channelFirst = first < second;
    const int attributes = channelFirst ? first : second;
    const int candidateCount = channelFirst ? second : first;
    const int classCount = attributes - 4;
    const float inverseScale = 1.0f / scale;
    auto valueAt = [&](int candidate, int attribute) {
        return channelFirst ? output[attribute * candidateCount + candidate]
                            : output[candidate * attributes + attribute];
    };

    std::vector<YoloBoundingBox> candidates;
    for (int candidate = 0; candidate < candidateCount; ++candidate)
    {
        int classId = 0;
        float score = 0.0f;
        for (int currentClass = 0; currentClass < classCount; ++currentClass)
        {
            if (valueAt(candidate, currentClass + 4) > score)
            {
                score = valueAt(candidate, currentClass + 4);
                classId = currentClass;
            }
        }
        if (score < CONF_THRESH || classId >= static_cast<int>(class_names.size()))
            continue;
        const float centerX = (valueAt(candidate, 0) - padX) * inverseScale;
        const float centerY = (valueAt(candidate, 1) - padY) * inverseScale;
        const float width = valueAt(candidate, 2) * inverseScale;
        const float height = valueAt(candidate, 3) * inverseScale;
        cv::Rect2f rect(centerX - width / 2.0f, centerY - height / 2.0f, width, height);
        rect &= cv::Rect2f(0, 0, static_cast<float>(image.cols), static_cast<float>(image.rows));
        candidates.emplace_back(rect, class_names[classId], score);
    }

    std::sort(candidates.begin(), candidates.end(), [](const YoloBoundingBox& firstBox,
                                                        const YoloBoundingBox& secondBox) {
        return firstBox.GetScore() > secondBox.GetScore();
    });
    for (const YoloBoundingBox& candidate : candidates)
    {
        bool suppressed = false;
        for (const YoloBoundingBox& selected : boxes)
            if (candidate.GetLabel() == selected.GetLabel() &&
                intersectionOverUnion(candidate.GetRect(), selected.GetRect()) > NMS_THRESH)
                suppressed = true;
        if (!suppressed)
        {
            if (candidate.GetLabel() == "person" && mpTracker)
                mpTracker->mvDynamicArea.push_back(candidate.GetRect());
            boxes.push_back(candidate);
        }
    }
}

void YoloDetector::Run()
{
    while (true)
    {
        usleep(1);
        if (!isNewImgArrived())
            continue;
        if (mImg.channels() == 1)
            cv::cvtColor(mImg, mImg, cv::COLOR_GRAY2RGB);
        std::unique_lock<std::mutex> resultLock(mMutexNewYoloDetector);
        mpTracker->yoloBoundingBoxList.clear();
        mpTracker->mvDynamicArea.clear();
        if (mbYOLO)
            DetectByTensorRT(mImg, mDepth, mpTracker->yoloBoundingBoxList);
        mpTracker->mbNewSegImgFlag = true;
        if (CheckFinish())
            break;
    }
}

void YoloDetector::SetTracker(Tracking* tracker) { mpTracker = tracker; }

bool YoloDetector::isNewImgArrived()
{
    std::unique_lock<std::mutex> lock(mMutexGetNewImg);
    if (!mbNewImgFlag)
        return false;
    mbNewImgFlag = false;
    return true;
}

YoloBoundingBox::YoloBoundingBox(cv::Rect2f inputRect, std::string inputLabel, float inputScore)
    : rect(inputRect), label(inputLabel), id(0), score(inputScore),
      width(inputRect.width), height(inputRect.height) {}

YoloBoundingBox::YoloBoundingBox(float x1, float y1, float x2, float y2,
                                 std::string inputLabel, float inputScore)
    : rect(x1, y1, x2 - x1, y2 - y1), label(inputLabel), id(0), score(inputScore),
      width(x2 - x1), height(y2 - y1) {}

bool YoloDetector::CheckFinish()
{
    std::unique_lock<std::mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void YoloDetector::RequestFinish()
{
    std::unique_lock<std::mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

}

#if 0
#define USE_FP16  // set USE_INT8 or USE_FP16 or USE_FP32
#define DEVICE 0  // GPU id
#define NMS_THRESH 0.4
#define CONF_THRESH 0.5
#define BATCH_SIZE 1
#define MAX_IMAGE_INPUT_SIZE_THRESH 3000 * 3000 // ensure it exceed the maximum size in the input images !
using namespace std;
// stuff we know about the network and the input/output blobs
static const int INPUT_H = Yolo::INPUT_H;
static const int INPUT_W = Yolo::INPUT_W;
static const int CLASS_NUM = Yolo::CLASS_NUM;
static const int OUTPUT_SIZE = Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) + 1;  // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
static Logger gLogger;

int inputIndex;
int outputIndex;
uint8_t* img_host = nullptr;
uint8_t* img_device = nullptr;
cudaStream_t stream;
int fcount = 0;
std::vector<cv::Mat> imgs_buffer(BATCH_SIZE);
float* buffers[2];
static float prob[BATCH_SIZE * OUTPUT_SIZE];

namespace ORB_SLAM2
{

YoloDetector::YoloDetector():cpuThreadNum(1), mbNewImgFlag(false),
    mbFinishRequested(false), mpTracker(NULL), mbTensorRT(true), mbYOLO(false),
    context(NULL)
{
    std::ifstream f("model/coco.names");
    std::string name = "";
    while (std::getline(f, name))
    {
        this->class_names.push_back(name);
    }

    std::string engine_name = "model/yolov5x.engine";
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engine_name << " error!" << std::endl;
    }
    char *trtModelStream = nullptr;
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();

    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    assert(engine->getNbBindings() == 2);
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc((void**)&buffers[inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));

    // Create stream
    CUDA_CHECK(cudaStreamCreate(&stream));

    // prepare input data cache in pinned memory
    CUDA_CHECK(cudaMallocHost((void**)&img_host, MAX_IMAGE_INPUT_SIZE_THRESH * 3));
    // prepare input data cache in device memory
    CUDA_CHECK(cudaMalloc((void**)&img_device, MAX_IMAGE_INPUT_SIZE_THRESH * 3));
    std::cout << "load model success" << std::endl;
}

void YoloDetector::DetectByTensorRT(cv::Mat& image, cv::Mat& depth1, std::vector<YoloBoundingBox>& yoloBoundingBoxList)
{
    
    cv::Mat img = image.clone();
    if (img.empty()) return;
    imgs_buffer[0] = img;
    float* buffer_idx = (float*)buffers[inputIndex];
    size_t  size_image = img.cols * img.rows * 3;
    size_t  size_image_dst = INPUT_H * INPUT_W * 3;
    //copy data to pinned memory
    memcpy(img_host,img.data,size_image);
    //copy data to device memory
    CUDA_CHECK(cudaMemcpyAsync(img_device,img_host,size_image,cudaMemcpyHostToDevice,stream));
    preprocess_kernel_img(img_device, img.cols, img.rows, buffer_idx, INPUT_W, INPUT_H, stream);
    // Run inference
    doInference(*context, stream, (void**)buffers, prob, BATCH_SIZE);
    auto end = std::chrono::system_clock::now();
    std::vector<std::vector<Yolo::Detection>> batch_res(1);
    auto& res = batch_res[0];
    nms(res, &prob[0], CONF_THRESH, NMS_THRESH);
    for(int i=0; i<res.size(); i++)
    {
        cv::Rect2f rect= get_rect(img,res[i].bbox);
        if(class_names[res[i].class_id]=="person")
            mpTracker->mvDynamicArea.push_back(rect);
        yoloBoundingBoxList.push_back(YoloBoundingBox(rect, class_names[res[i].class_id], res[i].conf));
    }
    cv::Mat depth = depth1.clone();
    //if(depth1.type()!=CV_32F)
    depth1.convertTo(depth, CV_32F, 1.0f/5000);
    // std::ofstream outFile("yolo_depth.txt");
    // if (!outFile) {
    //     std::cerr << "无法打cal文件。" << std::endl;
    //     return ;
    // }
    // for(int i=0; i<depth.rows; i++)
    // {
    //     for(int j=0; j<depth.cols; j++)
    //         outFile << depth.at<float>(i,j) <<endl;
    // }
    // outFile.close();
    // std::cout << "yolo_depth Saved!!" << std::endl;



    /***
    if(!yoloBoundingBoxList.empty())
    {
        for(auto it = yoloBoundingBoxList.begin(); it != yoloBoundingBoxList.end(); it++)
        {
            int id = it - yoloBoundingBoxList.begin();
            
            
            if(yoloBoundingBoxList[id].GetLabel()!="person")
                continue;
            //     it->box_semantic[0]

            cv::Rect2f tempRect = yoloBoundingBoxList[id].GetRect();
            cv::Point2f p1 = tempRect.tl();
            cv::Point2f p2 = tempRect.br();
            int x1 = (int) p1.x;
            int y1 = (int) p1.y;
            int x2 = (int) p2.x;
            int y2 = (int) p2.y;
            for(int i=x1; i<x2; i+=4)
            {
                for(int j=y1; j<y2; j+=4)
                {
                    
                    //cout<<"current vector size: "<<it->mvdep.size()<<endl;
                    if(i>=0 && i<depth.cols && j>=0 && j<depth.rows)
                    {
                        //cout<<"i: "<<i<<",j: "<<j<<",value: "<<depth.at<float>(j,i)<<endl;
                        if(depth.at<float>(j,i)<0 )
                            continue;
                        else if(depth.at<float>(j,i)>100)
                            it->mvdep.emplace_back(10);
                        it->mvdep.emplace_back(depth.at<float>(j,i));
                    }
                
                }
            } 
            int M = it->mvdep.size();
            sort(it->mvdep.begin(), it->mvdep.end());
            it->mvdep.erase(it->mvdep.begin()+0.7*M,it->mvdep.end());
            M = it->mvdep.size();
            it->dep_sum = std::accumulate(it->mvdep.begin(), it->mvdep.end(), 0.0);
            
            it->dep_average = it->dep_sum/M;
            float sum = 0.0;
            for(auto d:it->mvdep)
            {
                sum += (d - it->dep_average)*(d - it->dep_average);
            }
            it->dep_stdcov = std::sqrt(sum/(M));
            //int x = it-yoloBoundingBoxList.begin();
            //cout<<" it->dep_average: "<<it->dep_average<<" it->dep_stdcov: "<<it->dep_stdcov<<endl;
        }
    }***/
 

 
}

void YoloDetector::Run() {
    while (1)
    {
        
        usleep(1);
        if(!isNewImgArrived())
            continue;
        
        if(mImg.channels() == 1)
            cvtColor(mImg,mImg,cv::COLOR_GRAY2RGB);
        std::vector<YoloBoundingBox> yoloBoundingBoxLists;
      
        {
            // Publish detections and the completion flag under the same mutex.
            // Tracking can therefore never consume a partially filled mask.
            unique_lock<mutex> resultLock(mMutexNewYoloDetector);
            mpTracker->yoloBoundingBoxList.clear();
            mpTracker->mvDynamicArea.clear();
        if(mbYOLO)
        {
#ifdef TIMES
            std::chrono::steady_clock::time_point yolo1 = std::chrono::steady_clock::now();
#endif  cout<<"tensort take"<<endl;
                DetectByTensorRT(mImg,mDepth, mpTracker->yoloBoundingBoxList);

#ifdef TIMES
            std::chrono::steady_clock::time_point yolo2 = std::chrono::steady_clock::now();
            cout << "YOLO time:" << std::chrono::duration_cast<std::chrono::duration<double> >(yolo2 - yolo1).count() <<endl;
#endif
        }
            mpTracker->mbNewSegImgFlag=true;
        }
        if(CheckFinish())
            break;
    }
}

void YoloDetector::SetTracker(Tracking* pTracker) {
    mpTracker = pTracker;
}

bool YoloDetector::isNewImgArrived() {
    unique_lock<mutex> lock(mMutexGetNewImg);
    //cout<<"mbNewImgFlag: "<<mbNewImgFlag<<endl;
    if(mbNewImgFlag) {
        mbNewImgFlag = false;
        return true;
    }
    else
        return false;
}

YoloBoundingBox::YoloBoundingBox(cv::Rect2f input_rect, std::string input_label, float score){
    this->rect = input_rect;
    this->label = input_label;
    this->score = score;
    this->width = input_rect.br().x-input_rect.tl().x;
    this->height = input_rect.br().y-input_rect.tl().x;
}

YoloBoundingBox::YoloBoundingBox(float x1, float y1, float x2, float y2, std::string input_label, float score){
    cv::Point2f p1 = cv::Point2f(x1, y1);
    cv::Point2f p2 = cv::Point2f(x2, y2);
    this->rect = cv::Rect2f(p1, p2);
    this->label = input_label;
    this->score = score;
    this->width = p2.x - p1.x;
    this->height = p2.y - p1.y;
}

bool YoloDetector::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void YoloDetector::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested=true;
}

static int get_width(int x, float gw, int divisor = 8) {
    return int(ceil((x * gw) / divisor)) * divisor;
}

static int get_depth(int x, float gd) {
    if (x == 1) return 1;
    int r = round(x * gd);
    if (x * gd - int(x * gd) == 0.5 && (int(x * gd) % 2) == 0) {
        --r;
    }
    return std::max<int>(r, 1);
}

ICudaEngine* build_engine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, float& gd, float& gw, std::string& wts_name) {
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{ 3, INPUT_H, INPUT_W });
    assert(data);
    std::map<std::string, Weights> weightMap = loadWeights(wts_name);
    /* ------ yolov5 backbone------ */
    auto conv0 = convBlock(network, weightMap, *data,  get_width(64, gw), 6, 2, 1,  "model.0");
    assert(conv0);
    auto conv1 = convBlock(network, weightMap, *conv0->getOutput(0), get_width(128, gw), 3, 2, 1, "model.1");
    auto bottleneck_CSP2 = C3(network, weightMap, *conv1->getOutput(0), get_width(128, gw), get_width(128, gw), get_depth(3, gd), true, 1, 0.5, "model.2");
    auto conv3 = convBlock(network, weightMap, *bottleneck_CSP2->getOutput(0), get_width(256, gw), 3, 2, 1, "model.3");
    auto bottleneck_csp4 = C3(network, weightMap, *conv3->getOutput(0), get_width(256, gw), get_width(256, gw), get_depth(6, gd), true, 1, 0.5, "model.4");
    auto conv5 = convBlock(network, weightMap, *bottleneck_csp4->getOutput(0), get_width(512, gw), 3, 2, 1, "model.5");
    auto bottleneck_csp6 = C3(network, weightMap, *conv5->getOutput(0), get_width(512, gw), get_width(512, gw), get_depth(9, gd), true, 1, 0.5, "model.6");
    auto conv7 = convBlock(network, weightMap, *bottleneck_csp6->getOutput(0), get_width(1024, gw), 3, 2, 1, "model.7");
    auto bottleneck_csp8 = C3(network, weightMap, *conv7->getOutput(0), get_width(1024, gw), get_width(1024, gw), get_depth(3, gd), true, 1, 0.5, "model.8");
    auto spp9 = SPPF(network, weightMap, *bottleneck_csp8->getOutput(0), get_width(1024, gw), get_width(1024, gw), 5, "model.9");
    /* ------ yolov5 head ------ */
    auto conv10 = convBlock(network, weightMap, *spp9->getOutput(0), get_width(512, gw), 1, 1, 1, "model.10");

    auto upsample11 = network->addResize(*conv10->getOutput(0));
    assert(upsample11);
    upsample11->setResizeMode(ResizeMode::kNEAREST);
    upsample11->setOutputDimensions(bottleneck_csp6->getOutput(0)->getDimensions());

    ITensor* inputTensors12[] = { upsample11->getOutput(0), bottleneck_csp6->getOutput(0) };
    auto cat12 = network->addConcatenation(inputTensors12, 2);
    auto bottleneck_csp13 = C3(network, weightMap, *cat12->getOutput(0), get_width(1024, gw), get_width(512, gw), get_depth(3, gd), false, 1, 0.5, "model.13");
    auto conv14 = convBlock(network, weightMap, *bottleneck_csp13->getOutput(0), get_width(256, gw), 1, 1, 1, "model.14");

    auto upsample15 = network->addResize(*conv14->getOutput(0));
    assert(upsample15);
    upsample15->setResizeMode(ResizeMode::kNEAREST);
    upsample15->setOutputDimensions(bottleneck_csp4->getOutput(0)->getDimensions());

    ITensor* inputTensors16[] = { upsample15->getOutput(0), bottleneck_csp4->getOutput(0) };
    auto cat16 = network->addConcatenation(inputTensors16, 2);

    auto bottleneck_csp17 = C3(network, weightMap, *cat16->getOutput(0), get_width(512, gw), get_width(256, gw), get_depth(3, gd), false, 1, 0.5, "model.17");

    /* ------ detect ------ */
    IConvolutionLayer* det0 = network->addConvolutionNd(*bottleneck_csp17->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.0.weight"], weightMap["model.24.m.0.bias"]);
    auto conv18 = convBlock(network, weightMap, *bottleneck_csp17->getOutput(0), get_width(256, gw), 3, 2, 1, "model.18");
    ITensor* inputTensors19[] = { conv18->getOutput(0), conv14->getOutput(0) };
    auto cat19 = network->addConcatenation(inputTensors19, 2);
    auto bottleneck_csp20 = C3(network, weightMap, *cat19->getOutput(0), get_width(512, gw), get_width(512, gw), get_depth(3, gd), false, 1, 0.5, "model.20");
    IConvolutionLayer* det1 = network->addConvolutionNd(*bottleneck_csp20->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.1.weight"], weightMap["model.24.m.1.bias"]);
    auto conv21 = convBlock(network, weightMap, *bottleneck_csp20->getOutput(0), get_width(512, gw), 3, 2, 1, "model.21");
    ITensor* inputTensors22[] = { conv21->getOutput(0), conv10->getOutput(0) };
    auto cat22 = network->addConcatenation(inputTensors22, 2);
    auto bottleneck_csp23 = C3(network, weightMap, *cat22->getOutput(0), get_width(1024, gw), get_width(1024, gw), get_depth(3, gd), false, 1, 0.5, "model.23");
    IConvolutionLayer* det2 = network->addConvolutionNd(*bottleneck_csp23->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.2.weight"], weightMap["model.24.m.2.bias"]);

    auto yolo = addYoLoLayer(network, weightMap, "model.24", std::vector<IConvolutionLayer*>{det0, det1, det2});
    yolo->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*yolo->getOutput(0));
    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
#if defined(USE_FP16)
    config->setFlag(BuilderFlag::kFP16);
#elif defined(USE_INT8)
    std::cout << "Your platform support int8: " << (builder->platformHasFastInt8() ? "true" : "false") << std::endl;
assert(builder->platformHasFastInt8());
config->setFlag(BuilderFlag::kINT8);
Int8EntropyCalibrator2* calibrator = new Int8EntropyCalibrator2(1, INPUT_W, INPUT_H, "./coco_calib/", "int8calib.table", INPUT_BLOB_NAME);
config->setInt8Calibrator(calibrator);
#endif

    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {   
        cout<<"yolodetector 328"<<endl;
        free((void*)(mem.second.values));
    }

    return engine;
}

ICudaEngine* build_engine_p6(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, float& gd, float& gw, std::string& wts_name) {
    INetworkDefinition* network = builder->createNetworkV2(0U);
    // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{ 3, INPUT_H, INPUT_W });
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights(wts_name);

    /* ------ yolov5 backbone------ */
    auto conv0 = convBlock(network, weightMap, *data,  get_width(64, gw), 6, 2, 1,  "model.0");
    auto conv1 = convBlock(network, weightMap, *conv0->getOutput(0), get_width(128, gw), 3, 2, 1, "model.1");
    auto c3_2 = C3(network, weightMap, *conv1->getOutput(0), get_width(128, gw), get_width(128, gw), get_depth(3, gd), true, 1, 0.5, "model.2");
    auto conv3 = convBlock(network, weightMap, *c3_2->getOutput(0), get_width(256, gw), 3, 2, 1, "model.3");
    auto c3_4 = C3(network, weightMap, *conv3->getOutput(0), get_width(256, gw), get_width(256, gw), get_depth(6, gd), true, 1, 0.5, "model.4");
    auto conv5 = convBlock(network, weightMap, *c3_4->getOutput(0), get_width(512, gw), 3, 2, 1, "model.5");
    auto c3_6 = C3(network, weightMap, *conv5->getOutput(0), get_width(512, gw), get_width(512, gw), get_depth(9, gd), true, 1, 0.5, "model.6");
    auto conv7 = convBlock(network, weightMap, *c3_6->getOutput(0), get_width(768, gw), 3, 2, 1, "model.7");
    auto c3_8 = C3(network, weightMap, *conv7->getOutput(0), get_width(768, gw), get_width(768, gw), get_depth(3, gd), true, 1, 0.5, "model.8");
    auto conv9 = convBlock(network, weightMap, *c3_8->getOutput(0), get_width(1024, gw), 3, 2, 1, "model.9");
    auto c3_10 = C3(network, weightMap, *conv9->getOutput(0), get_width(1024, gw), get_width(1024, gw), get_depth(3, gd), true, 1, 0.5, "model.10");
    auto sppf11 = SPPF(network, weightMap, *c3_10->getOutput(0), get_width(1024, gw), get_width(1024, gw), 5, "model.11");

    /* ------ yolov5 head ------ */
    auto conv12 = convBlock(network, weightMap, *sppf11->getOutput(0), get_width(768, gw), 1, 1, 1, "model.12");
    auto upsample13 = network->addResize(*conv12->getOutput(0));
    assert(upsample13);
    upsample13->setResizeMode(ResizeMode::kNEAREST);
    upsample13->setOutputDimensions(c3_8->getOutput(0)->getDimensions());
    ITensor* inputTensors14[] = { upsample13->getOutput(0), c3_8->getOutput(0) };
    auto cat14 = network->addConcatenation(inputTensors14, 2);
    auto c3_15 = C3(network, weightMap, *cat14->getOutput(0), get_width(1536, gw), get_width(768, gw), get_depth(3, gd), false, 1, 0.5, "model.15");

    auto conv16 = convBlock(network, weightMap, *c3_15->getOutput(0), get_width(512, gw), 1, 1, 1, "model.16");
    auto upsample17 = network->addResize(*conv16->getOutput(0));
    assert(upsample17);
    upsample17->setResizeMode(ResizeMode::kNEAREST);
    upsample17->setOutputDimensions(c3_6->getOutput(0)->getDimensions());
    ITensor* inputTensors18[] = { upsample17->getOutput(0), c3_6->getOutput(0) };
    auto cat18 = network->addConcatenation(inputTensors18, 2);
    auto c3_19 = C3(network, weightMap, *cat18->getOutput(0), get_width(1024, gw), get_width(512, gw), get_depth(3, gd), false, 1, 0.5, "model.19");

    auto conv20 = convBlock(network, weightMap, *c3_19->getOutput(0), get_width(256, gw), 1, 1, 1, "model.20");
    auto upsample21 = network->addResize(*conv20->getOutput(0));
    assert(upsample21);
    upsample21->setResizeMode(ResizeMode::kNEAREST);
    upsample21->setOutputDimensions(c3_4->getOutput(0)->getDimensions());
    ITensor* inputTensors21[] = { upsample21->getOutput(0), c3_4->getOutput(0) };
    auto cat22 = network->addConcatenation(inputTensors21, 2);
    auto c3_23 = C3(network, weightMap, *cat22->getOutput(0), get_width(512, gw), get_width(256, gw), get_depth(3, gd), false, 1, 0.5, "model.23");

    auto conv24 = convBlock(network, weightMap, *c3_23->getOutput(0), get_width(256, gw), 3, 2, 1, "model.24");
    ITensor* inputTensors25[] = { conv24->getOutput(0), conv20->getOutput(0) };
    auto cat25 = network->addConcatenation(inputTensors25, 2);
    auto c3_26 = C3(network, weightMap, *cat25->getOutput(0), get_width(1024, gw), get_width(512, gw), get_depth(3, gd), false, 1, 0.5, "model.26");

    auto conv27 = convBlock(network, weightMap, *c3_26->getOutput(0), get_width(512, gw), 3, 2, 1, "model.27");
    ITensor* inputTensors28[] = { conv27->getOutput(0), conv16->getOutput(0) };
    auto cat28 = network->addConcatenation(inputTensors28, 2);
    auto c3_29 = C3(network, weightMap, *cat28->getOutput(0), get_width(1536, gw), get_width(768, gw), get_depth(3, gd), false, 1, 0.5, "model.29");

    auto conv30 = convBlock(network, weightMap, *c3_29->getOutput(0), get_width(768, gw), 3, 2, 1, "model.30");
    ITensor* inputTensors31[] = { conv30->getOutput(0), conv12->getOutput(0) };
    auto cat31 = network->addConcatenation(inputTensors31, 2);
    auto c3_32 = C3(network, weightMap, *cat31->getOutput(0), get_width(2048, gw), get_width(1024, gw), get_depth(3, gd), false, 1, 0.5, "model.32");

    /* ------ detect ------ */
    IConvolutionLayer* det0 = network->addConvolutionNd(*c3_23->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.33.m.0.weight"], weightMap["model.33.m.0.bias"]);
    IConvolutionLayer* det1 = network->addConvolutionNd(*c3_26->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.33.m.1.weight"], weightMap["model.33.m.1.bias"]);
    IConvolutionLayer* det2 = network->addConvolutionNd(*c3_29->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.33.m.2.weight"], weightMap["model.33.m.2.bias"]);
    IConvolutionLayer* det3 = network->addConvolutionNd(*c3_32->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.33.m.3.weight"], weightMap["model.33.m.3.bias"]);

    auto yolo = addYoLoLayer(network, weightMap, "model.33", std::vector<IConvolutionLayer*>{det0, det1, det2, det3});
    yolo->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*yolo->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
#if defined(USE_FP16)
    config->setFlag(BuilderFlag::kFP16);
#elif defined(USE_INT8)
    std::cout << "Your platform support int8: " << (builder->platformHasFastInt8() ? "true" : "false") << std::endl;
assert(builder->platformHasFastInt8());
config->setFlag(BuilderFlag::kINT8);
Int8EntropyCalibrator2* calibrator = new Int8EntropyCalibrator2(1, INPUT_W, INPUT_H, "./coco_calib/", "int8calib.table", INPUT_BLOB_NAME);
config->setInt8Calibrator(calibrator);
#endif

    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        cout<<"yolodetector 435"<<endl;
        free((void*)(mem.second.values));
    }

    return engine;
}


void YoloDetector::doInference(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* output, int batchSize) {
    // infer on the batch asynchronously, and DMA output back to host
    context.enqueue(batchSize, buffers, stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
}

}
#endif
