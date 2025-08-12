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

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "src/utils/func_register.h"

class OCRReisizeNormImg : public BaseProcessor {
 public:
  OCRReisizeNormImg(std::vector<int> rec_image_shape = {3, 48, 320},
                    std::vector<int> input_shape = {})
      : rec_image_shape_(rec_image_shape), input_shape_(input_shape){};
  absl::StatusOr<std::vector<cv::Mat>> Apply(
      std::vector<cv::Mat>& input, const void* param = nullptr) const override;
  absl::StatusOr<cv::Mat> Resize(cv::Mat& image) const;
  absl::StatusOr<cv::Mat> StaticResize(cv::Mat& image) const;
  absl::StatusOr<cv::Mat> ResizeNormImg(cv::Mat& image,
                                        float max_wh_ratio) const;
  static constexpr int MAX_IMG_W = 3200;

 private:
  std::vector<int> rec_image_shape_;
  std::vector<int> input_shape_;
};

class CTCLabelDecode {
 public:
  CTCLabelDecode(const std::vector<std::string>& character_list = {},
                 bool use_space_char = true);
  absl::StatusOr<std::vector<std::pair<std::string, float>>> Apply(
      const cv::Mat& preds) const;
  absl::StatusOr<std::pair<std::string, float>> Process(
      const cv::Mat& pred_data) const;
  absl::StatusOr<std::pair<std::string, float>> Decode(
      std::list<int>& text_index, std::list<float>& text_prob,
      bool is_remove_duplicate = false) const;
  void AddSpecialChar();

 private:
  std::vector<std::string> character_list_;
  bool use_space_char_;
  std::unordered_map<int, std::string> dict_;

  const std::vector<int> IGNORE_TOKEN = {0};
};
