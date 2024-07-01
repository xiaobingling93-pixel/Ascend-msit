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

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>
#include "nvjpeg.h"
#include "cuda_runtime_api.h"

static uint32_t g_yuvSizeAlignment = 3;
static uint32_t g_yuvSizeNum = 2;

static int DevMalloc(void **p, size_t s)
{
    return (int)cudaMalloc(p, s);
}

static int DevFree(void *p)
{
    return (int)cudaFree(p);
}

int main(int argc, const char *argv[])
{
    if ((argc < 4) || (argv[1] == nullptr)) {
        std::cerr << "[ERROR] Please input: " << argv[0] << " <image_path> <imageHeight> <imageWidth>" << std::endl;
        return 1;
    }
    std::string input_file_name = std::string(argv[1]);
    int heights = std::stoi(argv[2]);
    int widths = std::stoi(argv[3]);
    std::string outfile_path = "sample_nvjpeg.jpg";

    std::ifstream file(input_file_name.c_str(), std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    file.close();

    unsigned char* pBuffer = NULL;
    uint32_t jpegInBufferSize = widths * heights * g_yuvSizeAlignment / g_yuvSizeNum;
    cudaError_t ret = cudaMalloc((void **)&pBuffer, jpegInBufferSize);
    if (cudaSuccess != ret) {
        std::cerr << "cudaMalloc failed for component Y: " << cudaGetErrorString(ret) << std::endl;
        return 1;
    }
    cudaMemcpy(pBuffer, buffer.data(), size, cudaMemcpyHostToDevice);

    nvjpegHandle_t nvjpeg_handle;
    nvjpegDevAllocator_t dev_allocator = {&DevMalloc, &DevFree};
    nvjpegCreate(NVJPEG_BACKEND_DEFAULT, &dev_allocator, &nvjpeg_handle);

    // create image desc
    nvjpegImage_t imgdesc = {
        // pointer to start of Y plane, U plane, V plane
        {pBuffer, pBuffer + widths*heights, pBuffer + widths*heights + (widths*heights / 4)},
        // pitch of Y plane, pitch of U plane, pitch of V plane
        {(unsigned int)widths, (unsigned int)widths/2, (unsigned int)widths/2},
    };

    // sample input parameters
    nvjpegEncoderParams_t encode_params;
    nvjpegEncoderParamsCreate(nvjpeg_handle, &encode_params, NULL);
    nvjpegEncoderParamsSetQuality(encode_params, 100, NULL);

    nvjpegChromaSubsampling_t subsampling = NVJPEG_CSS_420;
    nvjpegEncoderParamsSetSamplingFactors(encode_params, subsampling, NULL);

    // Do encode
    nvjpegEncoderState_t encoder_state;
    nvjpegEncoderStateCreate(nvjpeg_handle, &encoder_state, NULL);
    nvjpegEncodeYUV(nvjpeg_handle, encoder_state, encode_params, &imgdesc, subsampling, widths, heights, NULL);

    // Get output to host
    std::vector<unsigned char> obuffer;
    size_t length;
    nvjpegEncodeRetrieveBitstream(nvjpeg_handle, encoder_state, NULL, &length, NULL);
    obuffer.resize(length);
    nvjpegEncodeRetrieveBitstream(nvjpeg_handle, encoder_state, obuffer.data(), &length, NULL);

    // save pic
    std::cout << "Writing JPEG file: " << outfile_path << ", length: " << length << std::endl;
    std::ofstream outputFile(outfile_path, std::ios::out | std::ios::binary);
    outputFile.write(reinterpret_cast<const char *>(obuffer.data()), static_cast<int>(length));

    cudaFree(pBuffer);
    nvjpegEncoderParamsDestroy(encode_params);
    nvjpegEncoderStateDestroy(encoder_state);
    nvjpegDestroy(nvjpeg_handle);
}
