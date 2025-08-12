#include <paddle_inference_api.h>

#include <opencv2/opencv.hpp>

#include "src/base/base_pipeline.h"
#include "src/pipelines/doc_preprocessor/pipeline.h"

#include <chrono>  
#include <iostream>

#include "src/utils/ilogger.h"
#include "src/pipelines/ocr/pipeline.h"


int main() {
    auto start = std::chrono::high_resolution_clock::now();
  OCRPipelineParams params;
  params.enable_mkldnn = true;

  BasePipeline* infer = new _OCRPipeline("/workspace/cpp_infer_refactor/models/", params);

  std::string input_str = "/workspace/PaddleX/pp_structure_v3_demo.png";
  // std::vector<std::string>  inputs =  {input_str,"/workspace/cpp_infer_refactor/detect_image/doc_test_rotated copy 4.jpg"};
  // std::vector<std::string> inputs = {"/workspace/cpp_infer_refactor/detect_image/doc_test_rotated.jpg","/workspace/cpp_infer_refactor/detect_image/doc_test_rotated copy.jpg"};

 

  auto outputs = infer->Predict(input_str);

  for (auto& output : outputs) {
    output->Print();
    output->SaveToImg("./output/");
    output->SaveToJson("./output/");
  }

  auto end = std::chrono::high_resolution_clock::now();
  double cost_ms = std::chrono::duration<double, std::milli>(end - start).count();
  std::cout << "Total predict cost: " << cost_ms << " ms" << std::endl;

  delete infer;
  return 0;
}