/*
 * Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
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

#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>

void resize_opencv(std::string imgFile) {
    cv::Mat img = cv::imread(imgFile, -1);
    cv::Size newSize = cv::Size(img.cols/2, img.rows/2);
    cv::Mat resizedImg(newSize, CV_8UC3);
    cv::resize(img, resizedImg, newSize, 0.0, 0.0, cv::INTER_LINEAR);
    cv::imwrite("./opencv_resize_output.jpg", resizedImg);
}

const int NUM_REQUIRED_ARGS = 2;

int main(int argc, char *argv[]) {
    if (argc != NUM_REQUIRED_ARGS) {
        std::cout << "Usage ./resize_opencv xxx.jpg" << std::endl;
        return 1;
    }
    resize_opencv(argv[1]);
    cv::dnn dnn("placeholder");
    return 0;
}