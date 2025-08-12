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

TextDetPredictor::TextDetPredictor(
    const std::string& model_dir, const std::string& device,
    const std::string& precision, const bool enable_mkldnn, int batch_size,
    const std::unordered_map<std::string, std::string>& config,

    int limit_side_len, const std::string& limit_type, float thresh,
    float box_thresh, float unclip_ratio, const std::vector<int>& input_shape,
    int max_side_limit)
    : BasePredictor(model_dir, device, precision, enable_mkldnn, batch_size,
                    config, "image"),
      limit_side_len_(limit_side_len),
      limit_type_(limit_type),
      thresh_(thresh),
      box_thresh_(box_thresh),
      unclip_ratio_(unclip_ratio),
      input_shape_(input_shape),
      max_side_limit_(max_side_limit) {
  Build();
};

TextDetPredictor::TextDetPredictor(const std::string& model_dir,
                                   const TextDetPredictorParams& params)
    : BasePredictor(model_dir, params.device, params.precision,
                    params.enable_mkldnn, params.batch_size, params.config,
                    "image"),
      params_(params),
      limit_side_len_(params.limit_side_len),
      limit_type_(params.limit_type),
      thresh_(params.thresh),
      box_thresh_(params.box_thresh),
      unclip_ratio_(params.unclip_ratio),
      input_shape_(params.input_shape),
      max_side_limit_(params.max_side_limit) {
  Build();
};

void TextDetPredictor::Build() {
  const auto& pre_tfs = config_.PreProcessOpInfo();
  Register<ReadImage>("Read", pre_tfs.at("DecodeImage.img_mode"));

  Register<DetResizeForTest>(
      "Resize", std::stoi(pre_tfs.at("DetResizeForTest.resize_long")));
  Register<NormalizeImage>("Normalize");
  Register<ToCHWImage>("ToCHW");
  Register<ToBatch>("ToBatch");
  infer_ptr_ = CreateStaticInfer();
  const auto& post_params = config_.PostProcessOpInfo();
  post_op_["DBPostProcess"] = std::unique_ptr<DBPostProcess>(
      new DBPostProcess(std::stof(post_params.at("PostProcess.thresh")),
                        std::stof(post_params.at("PostProcess.box_thresh")),
                        std::stoi(post_params.at("PostProcess.max_candidates")),
                        std::stof(post_params.at("PostProcess.unclip_ratio"))));
};

std::vector<std::unique_ptr<BaseCVResult>> TextDetPredictor::Process(
    std::vector<cv::Mat>& batch_data) {
  std::vector<cv::Mat> origin_image = {};
  origin_image.reserve(batch_data.size());
  for (const auto& mat : batch_data) {
    origin_image.push_back(mat.clone());
  }
  auto batch_raw_imgs = pre_op_.at("Read")->Apply(batch_data);
  if (!batch_raw_imgs.ok()) {
    INFOE(batch_raw_imgs.status().ToString().c_str());
  }
  std::vector<int> origin_shape = {batch_raw_imgs.value()[0].rows,
                                   batch_raw_imgs.value()[0].cols};
  DetResizeForTestParam resize_param;
  resize_param.limit_side_len = limit_side_len_;
  resize_param.limit_type = limit_type_;
  resize_param.max_side_limit = max_side_limit_;
  auto batch_imgs =
      pre_op_.at("Resize")->Apply(batch_raw_imgs.value(), &resize_param);
  if (!batch_imgs.ok()) {
    INFOE(batch_imgs.status().ToString().c_str());
  }
  auto batch_imgs_normalize =
      pre_op_.at("Normalize")->Apply(batch_imgs.value());
  if (!batch_imgs_normalize.ok()) {
    INFOE(batch_imgs_normalize.status().ToString().c_str());
  }

  auto batch_imgs_to_chw =
      pre_op_.at("ToCHW")->Apply(batch_imgs_normalize.value());
  if (!batch_imgs_to_chw.ok()) {
    INFOE(batch_imgs_to_chw.status().ToString().c_str());
  }
  auto batch_imgs_to_batch =
      pre_op_.at("ToBatch")->Apply(batch_imgs_to_chw.value());
  if (!batch_imgs_to_batch.ok()) {
    INFOE(batch_imgs_to_batch.status().ToString().c_str());
  }
  auto infer_result = infer_ptr_->Apply(batch_imgs_to_batch.value());
  if (!infer_result.ok()) {
    INFOE(infer_result.status().ToString().c_str());
  }
  auto db_result = post_op_.at("DBPostProcess")
                       ->Apply(infer_result.value()[0], origin_shape);

  if (!db_result.ok()) {
    INFOE(db_result.status().ToString().c_str());
  }

  std::vector<std::unique_ptr<BaseCVResult>> base_cv_result_ptr_vec = {};
  for (int i = 0; i < db_result.value().size(); i++, input_index_++) {
    TextDetPredictorResult predictor_result;
    if (!input_path_.empty()) {
      if (input_index_ == input_path_.size()) input_index_ = 0;
      predictor_result.input_path = input_path_[input_index_];
    }
    predictor_result.input_image = origin_image[i];
    predictor_result.dt_polys = db_result.value()[i].first;
    predictor_result.dt_scores = db_result.value()[i].second;
    predictor_result_vec_.push_back(predictor_result);
    base_cv_result_ptr_vec.push_back(
        std::unique_ptr<BaseCVResult>(new TextDetResult(predictor_result)));
  }

  return base_cv_result_ptr_vec;
}
