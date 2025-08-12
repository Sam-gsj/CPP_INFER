// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "src/base/base_pipeline.h"
#include "src/common/image_batch_sampler.h"
#include "src/common/processors.h"
#include "src/modules/image_classification/predictor.h"
#include "src/modules/text_detection/predictor.h"
#include "src/modules/text_recogntion/predictor.h"
#include "src/pipelines/doc_preprocessor/pipeline.h"
#include "src/utils/ilogger.h"

struct TextDetParams {
  int text_det_limit_side_len = -1;
  std::string text_det_limit_type = "";
  int text_det_max_side_limit = -1;
  float text_det_thresh = -1;
  float text_det_box_thresh = -1;
  float text_det_unclip_ratio = -1;
};

struct OCRPipelineResult {
  std::string input_path = "";
  DocPreprocessorPipelineResult doc_preprocessor_res;
  std::vector<std::vector<cv::Point2f>> dt_polys = {};
  std::unordered_map<std::string, bool> model_settings = {};
  TextDetParams text_det_params;
  std::string text_type = "";
  float text_rec_score_thresh = 0.0;
  std::vector<std::string> rec_texts = {};
  std::vector<float> rec_scores = {};
  std::vector<int> textline_orientation_angles = {};
  std::vector<std::vector<cv::Point2f>> rec_polys = {};
  std::vector<std::array<float, 4>> rec_boxes = {};
};

struct OCRPipelineParams {
  std::string device = "cpu";
  std::string precision = "fp32";
  bool enable_mkldnn = false;
  std::unordered_map<std::string, std::string> config = {};
  int textline_orientation_batch_size = 1;
  int text_recognition_batch_size = 1;
  bool use_doc_orientation_classify = true;
  bool use_doc_unwarping = true;
  bool use_textline_orientation = true;
  int text_det_limit_side_len = 64;
  std::string text_det_limit_type = "min";
  float text_det_thresh = 0.3;
  float text_det_box_thresh = 0.6;
  float text_det_unclip_ratio = 2.0;
  std::vector<int> text_det_input_shape = {};
  float text_rec_score_thresh = 0.0;
  std::vector<int> text_rec_input_shape = {};
  std::string lang = "";
};

class _OCRPipeline : public BasePipeline {
 public:
  explicit _OCRPipeline(const std::string& model_dir,
                        const OCRPipelineParams& params);
  virtual ~_OCRPipeline() = default;
  _OCRPipeline() = delete;

  std::vector<std::unique_ptr<BaseCVResult>> Predict(
      const std::vector<std::string>& input) override;

  std::vector<OCRPipelineResult> PipelineResult() const {
    return pipeline_result_vec_;
  };

  static absl::StatusOr<std::vector<cv::Mat>> RotateImage(
      const std::vector<cv::Mat>& image_array_list,
      const std::vector<int>& rotate_angle_list);

  std::unordered_map<std::string, bool> GetModelSettings() const;
  TextDetParams GetTextDetParams() const { return text_det_params_; };

 private:
  OCRPipelineParams params_;
  YamlConfig config_;
  std::unique_ptr<BaseBatchSampler> batch_sampler_ptr_;
  std::vector<OCRPipelineResult> pipeline_result_vec_;
  bool use_doc_preprocessor_ = false;
  bool use_doc_orientation_classify_ = false;
  bool use_doc_unwarping_ = false;
  std::unique_ptr<BasePipeline> doc_preprocessors_pipeline_;
  bool use_textline_orientation_ = false;
  std::unique_ptr<BasePredictor> textline_orientation_model_;
  std::unique_ptr<BasePredictor> text_det_model_;
  std::unique_ptr<BasePredictor> text_rec_model_;
  std::unique_ptr<CropByPolys> crop_by_polys_;
  std::function<std::vector<std::vector<cv::Point2f>>(
      const std::vector<std::vector<cv::Point2f>>&)>
      sort_boxes_;
  float text_rec_score_thresh_ = 0.0;
  std::string text_type_;
  TextDetParams text_det_params_;
};

class OCRPipeline
    : public AutoParallelSimpleInferencePipeline<
          _OCRPipeline, OCRPipelineParams, std::vector<std::string>,
          std::vector<std::unique_ptr<BaseCVResult>>> {
 public:
  OCRPipeline(const std::string& model_dir, const OCRPipelineParams& params,
              int thread_num = 1)
      : AutoParallelSimpleInferencePipeline(model_dir, params, thread_num),
        thread_num_(thread_num){};

  std::vector<std::unique_ptr<BaseCVResult>> Predict(
      const std::vector<std::string>& input) override;

 private:
  int thread_num_;
  std::unique_ptr<BaseBatchSampler> batch_sampler_ptr_;
};
