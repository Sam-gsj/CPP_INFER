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

#include "result.h"

#include <fstream>
#include <string>

#include "src/utils/utility.h"
#include "third_party/nlohmann/json.hpp"

using json = nlohmann::json;
void DocTrResult::SaveToImg(const std::string& save_path) {
  auto full_path = Utility::SmartCreateDirectoryForImage(
      save_path, predictor_result_.input_path);
  if (!full_path.ok()) {
    INFOE(full_path.status().ToString().c_str());
  }
  bool success = cv::imwrite(full_path.value(), predictor_result_.doctr_img);
  if (!success) {
    INFOE("Failed to write the image %s", full_path.value().c_str());
  }
}

void DocTrResult::Print() const {
  std::cout << "{\n  \"res\": {" << std::endl;

  std::cout << "    \"input_path\": {" << predictor_result_.input_path << "},"
            << std::endl;
  std::cout << "    \"doctr_img\": {"
            << "..."
            << "}" << std::endl;
  std::cout << "}" << std::endl;
}

void DocTrResult::SaveToJson(const std::string& save_path) const {
  nlohmann::ordered_json j;

  j["input_path"] = predictor_result_.input_path;
  j["page_index"] = nullptr;  //********

  nlohmann::json mat_array = nlohmann::json::array();

  for (int i = 0; i < predictor_result_.doctr_img.rows; ++i) {
    nlohmann::json row = nlohmann::json::array();
    for (int j = 0; j < predictor_result_.doctr_img.cols; ++j) {
      cv::Vec3b color = predictor_result_.doctr_img.at<cv::Vec3b>(i, j);
      row.push_back({color[0], color[1], color[2]});
    }
    mat_array.push_back(row);
  }
  j["doctr_img"] = mat_array;

  absl::StatusOr<std::string> full_path;
  if (predictor_result_.input_path.empty()) {
    INFOW("Input path is empty, will use output.jpg instead!");
    full_path = Utility::SmartCreateDirectoryForImage(save_path, "output.jpg");
  } else {
    full_path = Utility::SmartCreateDirectoryForImage(
        save_path, predictor_result_.input_path);
  }
  if (!full_path.ok()) {
    INFOE(full_path.status().ToString().c_str());
  }
  std::ofstream file(full_path.value());
  if (file.is_open()) {
    file << j.dump(4);
    file.close();
  } else {
    INFOE("Could not open file for writing: %s", save_path.c_str());
  }
}

int DocTrResult::getAdaptiveFontScale(const std::string& text, int imgWidth,
                                      int maxWidth, int minFont, int maxFont,
                                      int thickness, int& outBaseline,
                                      int& outFontFace) {
  int fontFace = cv::FONT_HERSHEY_SIMPLEX;
  double fontScale = 1.0;
  int baseline = 0;
  int bestFontSize = minFont;

  for (int fontSize = maxFont; fontSize >= minFont; --fontSize) {
    fontScale = fontSize / 20.0;
    int base;
    cv::Size textSize =
        cv::getTextSize(text, fontFace, fontScale, thickness, &base);
    if (textSize.width <= maxWidth) {
      bestFontSize = fontSize;
      outBaseline = base;
      outFontFace = fontFace;
      return fontScale;
    }
  }
  outBaseline = 0;
  outFontFace = fontFace;
  return minFont / 20.0;
}
