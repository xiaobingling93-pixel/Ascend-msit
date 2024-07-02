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

#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "acl/ops/acl_dvpp.h"


static int32_t g_deviceId = 0;
static aclrtContext g_context = nullptr;
static aclrtStream g_stream = nullptr;
static aclrtRunMode g_runMode;

static uint32_t g_yuvSizeAlignment = 3;
static uint32_t g_yuvSizeNum = 2;


static void Init()
{
    // ACL Init
    aclInit(nullptr);
    // resource manage
    aclrtSetDevice(g_deviceId);
    aclrtCreateContext(&g_context, g_deviceId);
    aclrtCreateStream(&g_stream);
    aclrtGetRunMode(&g_runMode);
}

int main(int argc, char *argv[])
{
    if ((argc < 4) || (argv[1] == nullptr)) {
        std::cerr << "[ERROR] Please input: " << argv[0] << " <image_path> <imageHeight> <imageWidth>" << std::endl;
        return 1;
    }
    std::string input_file_name = std::string(argv[1]);
    int heights = std::stoi(argv[2]);
    int widths = std::stoi(argv[3]);
    std::string outfile_path = "sample_acl_v1.jpg";

    Init();
    std::cout << "Open device " << g_deviceId << " success" << std::endl;

    std::ifstream file(input_file_name.c_str(), std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    file.close();

    unsigned char* pBuffer = NULL;
    uint32_t jpegInBufferSize = widths * heights * g_yuvSizeAlignment / g_yuvSizeNum;
    aclError ret = acldvppMalloc((void **)&pBuffer, jpegInBufferSize);
    if (ACL_SUCCESS != ret) {
        std::cerr << "acldvppMalloc malloc device data buffer failed, aclRet is " << ret << std::endl;
        return 1;
    }
    aclrtMemcpy(pBuffer, jpegInBufferSize, buffer.data(), jpegInBufferSize, ACL_MEMCPY_HOST_TO_DEVICE);

    // create dvpp channel
    acldvppChannelDesc *dvppChannelDesc = acldvppCreateChannelDesc();
    acldvppCreateChannel(dvppChannelDesc);

    // create image desc
    acldvppPicDesc *encodeInputDesc = acldvppCreatePicDesc();
    acldvppSetPicDescData(encodeInputDesc, reinterpret_cast<void *>(pBuffer));
    acldvppSetPicDescWidth(encodeInputDesc, widths);
    acldvppSetPicDescHeight(encodeInputDesc, heights);
    acldvppSetPicDescWidthStride(encodeInputDesc, widths);
    acldvppSetPicDescHeightStride(encodeInputDesc, heights);
    acldvppSetPicDescSize(encodeInputDesc, jpegInBufferSize);
    acldvppSetPicDescFormat(encodeInputDesc, PIXEL_FORMAT_YUV_SEMIPLANAR_420);

    acldvppJpegeConfig *jpegeConfig = acldvppCreateJpegeConfig();
    acldvppSetJpegeConfigLevel(jpegeConfig, 100);  // default optimal level (0-100)

    // output dev mem malloc
    void* encodeOutBufferDev;
    uint32_t length;
    acldvppJpegPredictEncSize(encodeInputDesc, jpegeConfig, &length);
    ret = acldvppMalloc(&encodeOutBufferDev, length);

    // Do encode
    acldvppJpegEncodeAsync(dvppChannelDesc, encodeInputDesc, encodeOutBufferDev, &length, jpegeConfig, g_stream);
    aclrtSynchronizeStream(g_stream);

    // Get output to host
    std::vector<unsigned char> oBuf(length);
    aclrtMemcpy(oBuf.data(), length, encodeOutBufferDev, length, ACL_MEMCPY_DEVICE_TO_HOST);

    // save pic
    std::cout << "Writing JPEG file: " << outfile_path << ", length: " << length << std::endl;
    std::ofstream outputFile(outfile_path, std::ios::out | std::ios::binary);
    outputFile.write(reinterpret_cast<const char *>(oBuf.data()), static_cast<int>(length));

    acldvppDestroyJpegeConfig(jpegeConfig);
    acldvppDestroyPicDesc(encodeInputDesc);
    acldvppFree(pBuffer);
    acldvppFree(encodeOutBufferDev);

    aclrtDestroyStream(g_stream);
    aclrtDestroyContext(g_context);
    aclrtResetDevice(g_deviceId);
    aclFinalize();
}
