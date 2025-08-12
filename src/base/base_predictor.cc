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

#include "base_predictor.h"

#include <yaml-cpp/yaml.h>

#include <iostream>

#include "base_batch_sampler.h"
#include "src/common/image_batch_sampler.h"
#include "src/utils/ilogger.h"
#include "src/utils/pp_option.h"
#include "src/utils/utility.h"

BasePredictor::BasePredictor(
    const std::string& model_dir, const std::string& device,
    const std::string& precision, const bool enable_mkldnn, int batch_size,
    const std::unordered_map<std::string, std::string>& config,
    const std::string sampler_type)
    : model_dir_(model_dir),
      batch_size_(batch_size),
      config_(config),
      sampler_type_(sampler_type) {
  if (config.empty()) {
    config_ = YamlConfig(model_dir_);
  }
  auto status_build = BuildBatchSampler();
  if (!status_build.ok()) {
    INFOE("Build sampler fail: %s", status_build.ToString().c_str());
  }
  auto model_name = config_.GetString(std::string("Global.model_name"));
  if (!model_name.ok()) {
    INFOE(model_name.status().ToString().c_str());
  }
  model_name_ = model_name.value();
  pp_option_ptr_.reset(new PaddlePredictorOption());

  size_t pos = device.find(':');
  std::string device_type = "";
  int device_id = 0;
  if (pos != std::string::npos) {
    device_type = device.substr(0, pos);
    device_id = std::stoi(device.substr(pos + 1));
  } else {
    device_type = device;
    device_id = 0;
  }
  auto result = pp_option_ptr_->SetDeviceType(device_type);
  if (!result.ok()) {
    INFOE("Failed to set device : %s", result.ToString().c_str());
    return;
  }
  result = pp_option_ptr_->SetDeviceId(device_id);
  if (!result.ok()) {
    INFOE("Failed to set device id: %s", result.ToString().c_str());
    return;
  }
  if (enable_mkldnn) {
    if (precision == "fp16") {
      INFOW(
          "When MKLDNN is enabled, FP16 precision is not supported.The "
          "computation will proceed with FP32 instead.");
    }
    result = pp_option_ptr_->SetRunMode("mkldnn");
    if (!result.ok()) {
      INFOE("Failed to set run mode: %s", result.ToString().c_str());
      return;
    }
  } else if (precision == "fp16") {
    if (precision == "fp16") {
      result = pp_option_ptr_->SetRunMode("paddle_fp16");
    }
    if (!result.ok()) {
      INFOE("Failed to set run mode: %s", result.ToString().c_str());
      return;
    }
  }
  INFO(pp_option_ptr_->DebugString().c_str());
}

std::vector<std::unique_ptr<BaseCVResult>> BasePredictor::Predict(
    const std::string& input) {
  std::vector<std::string> inputs = {input};
  return Predict(inputs);
}

const PaddlePredictorOption& BasePredictor::PPOption() {
  return *pp_option_ptr_;
}

void BasePredictor::SetBatchSize(int batch_size) { batch_size_ = batch_size; }

std::unique_ptr<PaddleInfer> BasePredictor::CreateStaticInfer() {
  return std::unique_ptr<PaddleInfer>(
      new PaddleInfer(model_name_, model_dir_, MODEL_FILE_PREFIX, PPOption()));
}

absl::Status BasePredictor::BuildBatchSampler() {
  if (SAMPLER_TYPE.count(sampler_type_) == 0) {
    return absl::InvalidArgumentError("Unsupported sampler type !");
  } else if (sampler_type_ == "image") {
    batch_sampler_ptr_ =
        std::unique_ptr<BaseBatchSampler>(new ImageBatchSampler(batch_size_));
  }
  return absl::OkStatus();
}

const std::unordered_set<std::string> BasePredictor::SAMPLER_TYPE = {
    "image",
};
