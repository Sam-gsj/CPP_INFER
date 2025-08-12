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

#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "src/base/base_pipeline.h"
#include "src/common/image_batch_sampler.h"
#include "src/common/parallel.h"
#include "src/common/processors.h"
#include "src/utils/ilogger.h"
#include "src/utils/utility.h"

struct DocPreprocessorPipelineResult {
  std::string input_path = "";
  cv::Mat input_image;
  std::unordered_map<std::string, bool> model_settings;
  int angle = 0;
  cv::Mat rotate_image;
  cv::Mat output_image;
  cv::Mat image_all;
};

struct DocPreprocessorPipelineParams {
  std::string device = "cpu";
  std::string precision = "fp32";
  bool enable_mkldnn = false;
  std::unordered_map<std::string, std::string> config = {};
  bool use_doc_orientation_classify = false;
  bool use_doc_unwarping = false;
};

class _DocPreprocessorPipeline : public BasePipeline {
 public:
  explicit _DocPreprocessorPipeline(
      const std::string& model_dir, std::string device = "cpu",
      const std::string& precision = "fp32", bool enable_mkldnn = false,
      const std::unordered_map<std::string, std::string>& config = {},
      bool use_doc_orientation_classify = false,
      bool use_doc_unwarping = false);
  explicit _DocPreprocessorPipeline(
      const std::string& model_dir,
      const DocPreprocessorPipelineParams& params);
  virtual ~_DocPreprocessorPipeline() = default;

  _DocPreprocessorPipeline() = delete;

  std::vector<std::unique_ptr<BaseCVResult>> Predict(
      const std::vector<std::string>& input) override;

  std::unordered_map<std::string, bool> GetModelSettings(
      absl::optional<bool> use_doc_orientation_classify = absl::nullopt,
      absl::optional<bool> use_doc_unwarping = absl::nullopt) const;
  absl::Status CheckModelSettingsVaild(
      std::unordered_map<std::string, bool> model_settings) const;

  std::vector<DocPreprocessorPipelineResult> PipelineResult() const {
    return pipeline_result_vec_;
  };

 private:
  bool use_doc_orientation_classify_;
  bool use_doc_unwarping_;
  std::unique_ptr<BasePredictor> doc_ori_classify_model_;
  std::unique_ptr<BasePredictor> doc_unwarping_model_;
  DocPreprocessorPipelineParams params_;
  YamlConfig config_;
  std::unique_ptr<BaseBatchSampler> batch_sampler_ptr_;
  std::vector<DocPreprocessorPipelineResult> pipeline_result_vec_;
};

class DocPreprocessorPipeline
    : public AutoParallelSimpleInferencePipeline<
          _DocPreprocessorPipeline, DocPreprocessorPipelineParams,
          std::vector<std::string>,
          std::vector<std::unique_ptr<BaseCVResult>>> {
 public:
  DocPreprocessorPipeline(const std::string& model_dir,
                          const DocPreprocessorPipelineParams& params,
                          int thread_num = 1)
      : AutoParallelSimpleInferencePipeline(model_dir, params, thread_num),
        thread_num_(thread_num){};

  std::vector<std::unique_ptr<BaseCVResult>> Predict(
      const std::vector<std::string>& input) override;

 private:
  int thread_num_;
  std::unique_ptr<BaseBatchSampler> batch_sampler_ptr_;
};
