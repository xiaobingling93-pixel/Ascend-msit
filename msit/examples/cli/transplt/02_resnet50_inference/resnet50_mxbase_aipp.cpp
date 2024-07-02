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
    if (ret != APP_ERR_OK) {
        return ret;
    }

    uint32_t deviceId = 0;
    std::string modelPath = "resnet50_aipp.om";
    MxBase::Model net(modelPath, deviceId);

    auto start = std::chrono::high_resolution_clock::now();
    std::string imgFile = "test.png";
    MxBase::ImageProcessor processor;
    MxBase::Image decodedImage;
    processor.Decode(imgFile, decodedImage, MxBase::ImageFormat::RGB_888);
    auto end = std::chrono::high_resolution_clock::now();
    float timeDuration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1.0e6;
    std::cout << "Image process time duration: " << timeDuration << "ms" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    MxBase::Tensor tensor = decodedImage.ConvertToTensor();
    tensor.ToDevice(deviceId);
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
