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

struct WarpPredictorParams {
  std::string device = "cpu";
  std::string precision = "fp32";
  bool enable_mkldnn = false;
  int batch_size = 1;
  std::unordered_map<std::string, std::string> config = {};
};

struct WarpPredictorResult {
  std::string input_path = "";
  cv::Mat input_image;
  cv::Mat doctr_img;
};

class WarpPredictor : public BasePredictor {
 public:
  explicit WarpPredictor(
      const std::string& model_dir, const std::string& device = "cpu",
      const std::string& precision = "fp32", const bool enable_mkldnn = false,
      int batch_size = 1,
      const std::unordered_map<std::string, std::string>& config = {});
  explicit WarpPredictor(const std::string& model_dir,
                         const WarpPredictorParams& params);

  void Build();

  std::vector<std::unique_ptr<BaseCVResult>> Process(
      std::vector<cv::Mat>& batch_data) override;

  std::vector<WarpPredictorResult> PredictorResult() const {
    return predictor_result_vec_;
  };

  void ResetResult() override { predictor_result_vec_.clear(); };

 private:
  std::unordered_map<std::string, std::unique_ptr<DocTrPostProcess>> post_op_;
  std::vector<WarpPredictorResult> predictor_result_vec_;
  std::unique_ptr<PaddleInfer> infer_ptr_;
  WarpPredictorParams params_;
  int input_index_ = 0;
};
