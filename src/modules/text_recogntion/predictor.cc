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

#include "predictor.h"

#include "result.h"
#include "src/common/image_batch_sampler.h"

TextRecPredictor::TextRecPredictor(
    const std::string& model_dir, const std::string& device,
    const std::string& precision, const bool enable_mkldnn, int batch_size,
    const std::unordered_map<std::string, std::string>& config)
    : BasePredictor(model_dir, device, precision, enable_mkldnn, batch_size,
                    config, "image") {
  Build();
};

TextRecPredictor::TextRecPredictor(const std::string& model_dir,
                                   const TextRecPredictorParams& params)
    : BasePredictor(model_dir, params.device, params.precision,
                    params.enable_mkldnn, params.batch_size, params.config,
                    "image"),
      params_(params) {
  Build();
};

void TextRecPredictor::Build() {
  const auto& pre_params = config_.PreProcessOpInfo();
  Register<ReadImage>("Read", "BGR");
  Register<OCRReisizeNormImg>("ReisizeNorm");
  Register<ToBatch>("ToBatch");

  infer_ptr_ = CreateStaticInfer();
  const auto& post_params = config_.PostProcessOpInfo();
  post_op_["CTCLabelDecode"] = std::unique_ptr<CTCLabelDecode>(
      new CTCLabelDecode(YamlConfig::SmartParseVector(
                             post_params.at("PostProcess.character_dict"))
                             .vec_string));
};

std::vector<std::unique_ptr<BaseCVResult>> TextRecPredictor::Process(
    std::vector<cv::Mat>& batch_data) {
  std::vector<cv::Mat> origin_image = {};
  origin_image.reserve(batch_data.size());
  for (const auto& mat : batch_data) {
    origin_image.push_back(mat.clone());
  }
  auto batch_read = pre_op_.at("Read")->Apply(batch_data);
  if (!batch_read.ok()) {
    INFOE(batch_read.status().ToString().c_str());
  }

  auto batch_resize_norm = pre_op_.at("ReisizeNorm")->Apply(batch_read.value());
  if (!batch_resize_norm.ok()) {
    INFOE(batch_resize_norm.status().ToString().c_str());
  }

  auto batch_tobatch = pre_op_.at("ToBatch")->Apply(batch_resize_norm.value());
  if (!batch_tobatch.ok()) {
    INFOE(batch_tobatch.status().ToString().c_str());
  }
  auto batch_infer = infer_ptr_->Apply(batch_tobatch.value());
  if (!batch_infer.ok()) {
    INFOE(batch_infer.status().ToString().c_str());
  }

  auto ctc_result =
      post_op_.at("CTCLabelDecode")->Apply(batch_infer.value()[0]);

  if (!ctc_result.ok()) {
    INFOE(ctc_result.status().ToString().c_str());
  }

  std::vector<std::unique_ptr<BaseCVResult>> base_cv_result_ptr_vec = {};
  for (int i = 0; i < ctc_result.value().size(); i++, input_index_++) {
    TextRecPredictorResult predictor_result;
    if (!input_path_.empty()) {
      if (input_index_ == input_path_.size()) input_index_ = 0;
      predictor_result.input_path = input_path_[input_index_];
    }
    predictor_result.input_image = origin_image[i];
    predictor_result.rec_text = ctc_result.value()[i].first;
    predictor_result.rec_score = ctc_result.value()[i].second;
    predictor_result.vis_font =
        "/workspace/cpp_infer_refactor/models/PP-OCRv5_server_rec/simfang.ttf";
    predictor_result_vec_.push_back(predictor_result);
    base_cv_result_ptr_vec.push_back(
        std::unique_ptr<BaseCVResult>(new TextRecResult(predictor_result)));
  }
  return base_cv_result_ptr_vec;
}
