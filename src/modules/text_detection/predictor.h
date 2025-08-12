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

#include "processors.h"
#include "src/base/base_batch_sampler.h"
#include "src/base/base_cv_result.h"
#include "src/base/base_predictor.h"
#include "src/common/processors.h"

struct TextDetPredictorResult {
  std::string input_path = "";
  cv::Mat input_image;
  std::vector<std::vector<cv::Point2f>> dt_polys = {};
  std::vector<float> dt_scores = {};
};

struct TextDetPredictorParams {
  std::string device = "cpu";
  std::string precision = "fp32";
  bool enable_mkldnn = false;
  int batch_size = 1;
  std::unordered_map<std::string, std::string> config = {};
  int limit_side_len = 64;
  std::string limit_type = "min";
  float thresh = 0.3;
  float box_thresh = 0.6;
  float unclip_ratio = 1.5;
  std::vector<int> input_shape = {};
  int max_side_limit = 4000;
};

class TextDetPredictor : public BasePredictor {
 public:
  TextDetPredictor(  //*********************
      const std::string &model_dir, const std::string &device = "cpu",
      const std::string &precision = "fp32", const bool enable_mkldnn = false,
      int batch_size = 1,
      const std::unordered_map<std::string, std::string> &config = {},

      int limit_side_len = -1, const std::string &limit_type = "",
      float thresh = -1, float box_thresh = -1, float unclip_ratio = -1,
      const std::vector<int> &input_shape = std::vector<int>(),
      int max_side_limit = 4000);

  TextDetPredictor(const std::string &model_dir,
                   const TextDetPredictorParams &params);

  std::vector<TextDetPredictorResult> PredictorResult() const {
    return predictor_result_vec_;
  };

  void ResetResult() override { predictor_result_vec_.clear(); };

  void Build();

  std::vector<std::unique_ptr<BaseCVResult>> Process(
      std::vector<cv::Mat> &batch_data) override;

 private:
  int limit_side_len_;
  std::string limit_type_;
  float thresh_;
  float box_thresh_;
  float unclip_ratio_;
  std::vector<int> input_shape_;
  int max_side_limit_;

  std::unordered_map<std::string, std::unique_ptr<DBPostProcess>> post_op_;
  std::vector<TextDetPredictorResult> predictor_result_vec_;
  std::unique_ptr<PaddleInfer> infer_ptr_;
  TextDetPredictorParams params_;
  int input_index_ = 0;
};
