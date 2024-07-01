/*
 * Copyright (c) 2023-2023 Huawei Technologies Co., Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <opencv2/opencv.hpp>
#include "MxBase/MxBase.h"


int main(int argc, char **argv)
{
    APP_ERROR ret = MxBase::MxInit();
    if (ret != APP_ERR_OK) return ret;

    uint32_t deviceId = 0;
    std::string modelPath = "resnet50.om";
    MxBase::Model net(modelPath, deviceId);

    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat image = cv::imread("test.png", 1);
    if (image.empty()) {
        return 0;
    }

    int inputHeight = 224;
    int inputWidth = 224;
    auto meanChannel = cv::Scalar(123.675, 116.28, 103.53);
    auto stdChannel = cv::Scalar(58.395, 57.12, 57.375);
    cv::Mat resizedImage;
    cv::Mat blob;
    cv::resize(image, resizedImage, cv::Size(inputHeight, inputWidth));
    cv::dnn::blobFromImage(resizedImage, blob, 1.0, cv::Size(inputHeight, inputWidth), cv::Scalar(), true, false);
    blob = (blob - meanChannel) / stdChannel;

    auto end = std::chrono::high_resolution_clock::now();
    float timeDuration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1.0e6;
    std::cout << "Image process time duration: " << timeDuration << "ms" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    const std::vector<uint32_t> shape = {1, 3, 224, 224};
    MxBase::Tensor tensor = MxBase::Tensor((void *)blob.data, shape, MxBase::TensorDType::FLOAT32, deviceId);
    std::vector<MxBase::Tensor> mxInputs = {tensor};
    std::vector<MxBase::Tensor> outputs = net.Infer(mxInputs);
    outputs[0].ToHost();

    end = std::chrono::high_resolution_clock::now();
    timeDuration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1.0e6;
    std::cout << "Model inference time duration: " << timeDuration << "ms" << std::endl;

    int totalClasses = 1000;
    int argmax = 0;
    float maxScore = 0;
    float *data = (float *)outputs[0].GetData();
    for (int ii = 0; ii < totalClasses; ii++) {
        if (data[ii] > maxScore) {
            maxScore = data[ii];
            argmax = ii;
        }
    }
    std::cout << "index: " << argmax << std::endl;
    std::cout << "score: " << maxScore << std::endl;
    
    return 0;
}
