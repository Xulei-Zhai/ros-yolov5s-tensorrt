#include <ros/ros.h>
#include<sensor_msgs/Image.h>
#include<cv_bridge/cv_bridge.h>
#include<opencv2/opencv.hpp>

#include "cuda_utils.h"
#include "logging.h"
#include "utils.h"
#include "preprocess.h"
#include "postprocess.h"
#include "model.h"
#include <iostream>
#include <chrono>
#include <cmath>

using namespace nvinfer1;
static Logger gLogger;
const static int kOutputSize = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;



cudaStream_t stream;
IRuntime* runtime = nullptr;
ICudaEngine* engine = nullptr;
IExecutionContext* context = nullptr;
// Prepare cpu and gpu buffers
float* gpu_buffers[2];
float* cpu_output_buffer = nullptr;
char* serialized_engine ;
auto temp_time = std::chrono::system_clock::now();
float inference_fps =0.0; //frame per second
float stream_fps =0.0; //frame per second


void prepare_buffers(ICudaEngine* engine, float** gpu_input_buffer, float** gpu_output_buffer, float** cpu_output_buffer) {
  // getNbBindings() 返回一个整数值，表示在TensorRT推理引擎中绑定的张量数量。
  //绑定张量是指在推理过程中需要显式提供或接收数据的张量，通常包括输入张量和输出张量。
  assert(engine->getNbBindings() == 2);
  // In order to bind the buffers, we need to know the names of the input and output tensors.
  // Note that indices are guaranteed to be less than IEngine::getNbBindings()

  //使用getBindingIndex函数获取指定名称（kInputTensorName）的输入张量在推理引擎中的索引。
  //该函数返回一个整数值，表示输入张量的索引，并将其赋值给inputIndex变量。
  const int inputIndex = engine->getBindingIndex(kInputTensorName);
  const int outputIndex = engine->getBindingIndex(kOutputTensorName);
  assert(inputIndex == 0);
  assert(outputIndex == 1);
  // Create GPU buffers on device

  // void*是一种通用的方式来表示未指定类型的指针
  // 使用void*类型的指针的好处是它可以被隐式地转换为其他类型的指针。
  //这样就允许在运行时将void*指针转换为任意类型的指针，便于在后续的操作中使用。
  CUDA_CHECK(cudaMalloc((void**)gpu_input_buffer, kBatchSize * 3 * kInputH * kInputW * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)gpu_output_buffer, kBatchSize * kOutputSize * sizeof(float)));

  *cpu_output_buffer = new float[kBatchSize * kOutputSize];
}

void infer(IExecutionContext& context, cudaStream_t& stream, void** gpu_buffers, float* output, int batchsize) {
  context.enqueue(batchsize, gpu_buffers, stream, nullptr);
  CUDA_CHECK(cudaMemcpyAsync(output, gpu_buffers[1], batchsize * kOutputSize * sizeof(float), cudaMemcpyDeviceToHost, stream));
  cudaStreamSynchronize(stream);
}

void deserialize_engine(std::string& engine_name, IRuntime** runtime, ICudaEngine** engine, IExecutionContext** context) {
  std::ifstream file(engine_name, std::ios::binary);
  if (!file.good()) {
    std::cerr << "read " << engine_name << " error!" << std::endl;
    assert(false);
  }
  size_t size = 0;
  file.seekg(0, file.end);
  size = file.tellg();
  file.seekg(0, file.beg);
//   char* serialized_engine = new char[size];
  serialized_engine = new char[size];
  assert(serialized_engine);
  file.read(serialized_engine, size);  // 这行代码用于从文件中读取数据，并将其存储到名为serialized_engine的缓冲区中,size是要读取的字节数。
  file.close();  // 关闭文件流是一种良好的做法，它确保在不再需要访问文件时，释放相关资源并维护系统的一致状态。

// 这行代码调用createInferRuntime函数来创建一个TensorRT运行时对象，并将其赋值给指针*runtime。
//gLogger是一个用于记录日志的对象，用于输出TensorRT的运行时消息和错误信息。
  *runtime = createInferRuntime(gLogger);
  assert(*runtime);
  // 这行代码使用deserializeCudaEngine函数从序列化的引擎数据中反序列化出一个TensorRT引擎
  //，并将其赋值给指针*engine。serialized_engine是之前从文件中读取的引擎数据，size是引擎数据的大小。
  *engine = (*runtime)->deserializeCudaEngine(serialized_engine, size);
  assert(*engine);
  // 这行代码使用引擎对象的createExecutionContext函数创建一个TensorRT执行上下文（Execution Context）
  //，并将其赋值给指针*context。
  *context = (*engine)->createExecutionContext();
  assert(*context);
  // 这行代码使用delete[]操作符释放之前分配的serialized_engine缓冲区，以避免内存泄漏。
}

void a_main(cv::Mat& input_image ) {
    std::vector<cv::Mat> img_batch;
    img_batch.push_back(input_image);

    // Preprocess
    cuda_batch_preprocess(img_batch, gpu_buffers[0], kInputW, kInputH, stream);

    // Run inference
    auto start = std::chrono::system_clock::now();
    infer(*context, stream, (void**)gpu_buffers, cpu_output_buffer, kBatchSize);
    auto end = std::chrono::system_clock::now();
    std::cout << "inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    // NMS
    auto start1 = std::chrono::system_clock::now();
    std::vector<std::vector<Detection>> res_batch;
    batch_nms(res_batch, cpu_output_buffer, img_batch.size(), kOutputSize, kConfThresh, kNmsThresh);
    auto end1 = std::chrono::system_clock::now();
    // std::cout << "NMS time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count() << "ms" << std::endl;

    // Draw bounding boxes
    auto start2 = std::chrono::system_clock::now();
    draw_bbox(img_batch, res_batch);
    auto end2 = std::chrono::system_clock::now();
    // std::cout << "draw_bbox time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2).count() << "ms" << std::endl;
    // std::cout << "infer+NMS+draw_bbox- time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2 +end1 - start1 +end - start).count() << "ms" << std::endl;
}
    
void destory_engine(cudaStream_t stream, float** gpu_buffers, float* cpu_output_buffer, IExecutionContext* context, ICudaEngine* engine,  IRuntime* runtime){
     
     
  delete[] serialized_engine;  
     
    // Release stream and buffers
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(gpu_buffers[0]));
    CUDA_CHECK(cudaFree(gpu_buffers[1]));
    delete[] cpu_output_buffer;
    cuda_preprocess_destroy();
    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();
}



void imageCallback(const sensor_msgs::Image::ConstPtr msg){
    cv::Mat image;
    try{  
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        image = cv_ptr->image.clone();
    }
    catch(cv_bridge::Exception& e){  
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }    
    auto start = std::chrono::system_clock::now();
    
    a_main(image);
    auto end = std::chrono::system_clock::now();
    // std::cout<<inference_fps<<std::endl;
    inference_fps  = 0.5 * (inference_fps + 1000*1.0f/ std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
    stream_fps = 0.5*(stream_fps +1000*1.0f/std::chrono::duration_cast<std::chrono::milliseconds>(start - temp_time).count() ); 
    temp_time = start;
    std::stringstream ss, ss1;
    ss << "inference_fps: " << inference_fps;
    ss1<< "stream_fps:" << stream_fps;
    std::string text = ss.str();
    std::string text1 = ss1.str();
     // 在图像上输入文字
    cv::Point org(50, 75);  // 文字的起始坐标
    cv::Point org1(50, 50);  // 文字的起始坐标
    int fontFace = cv::FONT_HERSHEY_SIMPLEX;  // 字体类型
    double fontScale = 1.0;  // 字体缩放比例
    // cv::Scalar color(192, 192, 192);  // 银色（灰色）
    cv::Scalar color(0, 165, 255);  // 橙色
    
    int thickness = 2;  // 文字线宽
    cv::putText(image, text, org, fontFace, fontScale, color, thickness);
    cv::putText(image, text1, org1, fontFace, fontScale, color, thickness);
    cv::imshow("image", image);
    cv::waitKey(50);
    cv::destroyAllWindows;

    // sensor_msgs::ImagePtr processed_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg();
}



int main(int argc, char **argv){
    cudaSetDevice(kGpuId);
    // std::string engine_name = "/home/xx/xulei/catkin_ws/src/detect/engine/best.engine";
    std::string engine_name = "/home/xx/xulei/catkin_ws/src/detect/engine/yolov5s.engine";
    // Deserialize the engine from file

    deserialize_engine(engine_name, &runtime, &engine, &context);

    CUDA_CHECK(cudaStreamCreate(&stream));

    // Init CUDA preprocessing
    cuda_preprocess_init(kMaxInputImageSize);

    // // Prepare cpu and gpu buffers
    // float* gpu_buffers[2];
    // float* cpu_output_buffer = nullptr;
    prepare_buffers(engine, &gpu_buffers[0], &gpu_buffers[1], &cpu_output_buffer);

    ros::init(argc, argv, "image_subcriber");
    ros::NodeHandle nh;
    ros::Subscriber image_subscribe = nh.subscribe<sensor_msgs::Image>("/usb_cam/image_raw", 10, imageCallback );
    ros::Publisher image_publish = nh.advertise<sensor_msgs::Image>("/image_detected",10);
    ros::spin();

    destory_engine(stream, gpu_buffers, cpu_output_buffer,  context, engine, runtime);
    return 0;
}

