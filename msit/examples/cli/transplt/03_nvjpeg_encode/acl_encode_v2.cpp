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
#include "acl/acl_rt.h"
#include "acl/dvpp/hi_dvpp.h"


static int32_t g_deviceId = 0;
static aclrtContext g_context = nullptr;
static aclrtStream g_stream = nullptr;
static aclrtRunMode g_runMode;

static uint32_t g_yuv_sizeAlignment = 3;
static uint32_t g_yuv_sizeNum = 2;


static void Init()
{
    // ACL Init
    aclInit(nullptr);
    // resource manage
    aclrtSetDevice(g_deviceId);
    aclrtCreateContext(&g_context, g_deviceId);
    aclrtGetCurrentContext(&g_context);
    aclrtCreateStream(&g_stream);
    aclrtGetRunMode(&g_runMode);
    hi_mpi_sys_init();
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
    std::string outfile_path = "sample_acl_v2.jpg";

    Init();
    std::cout << "Open device " << g_deviceId << " success" << std::endl;

    std::ifstream file(input_file_name.c_str(), std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    file.close();

    unsigned char* pBuffer = NULL;
    uint32_t jpegInBufferSize = widths * heights * g_yuv_sizeAlignment / g_yuv_sizeNum;
    aclError ret = hi_mpi_dvpp_malloc(0, (void **)&pBuffer, jpegInBufferSize);
    if (ACL_SUCCESS != ret) {
        std::cerr << "acldvppMalloc malloc device data buffer failed, aclRet is " << ret << std::endl;
        return 1;
    }
    aclrtMemcpy(pBuffer, jpegInBufferSize, buffer.data(), jpegInBufferSize, ACL_MEMCPY_HOST_TO_DEVICE);

    // create encode channel
    hi_s32 g_chn_id = 0;
    hi_venc_chn_attr vencChnAttr;
    vencChnAttr.venc_attr.profile = 0;
    vencChnAttr.venc_attr.type = HI_PT_JPEG;
    vencChnAttr.venc_attr.pic_width = widths;
    vencChnAttr.venc_attr.pic_height = heights;
    vencChnAttr.venc_attr.max_pic_width = widths;
    vencChnAttr.venc_attr.max_pic_height = heights;
    vencChnAttr.venc_attr.is_by_frame = HI_TRUE; // get stream mode is field mode or frame mode
    vencChnAttr.venc_attr.buf_size = jpegInBufferSize;
    hi_mpi_venc_create_chn(g_chn_id, &vencChnAttr);

    // start venc chn
    hi_venc_start_param recvParam;
    recvParam.recv_pic_num = -1; // unspecified frame count
    hi_mpi_venc_start_chn(g_chn_id, &recvParam);

    // create image desc
    hi_video_frame_info inputFrame{};
    inputFrame.v_frame.virt_addr[0] = pBuffer;
    inputFrame.v_frame.virt_addr[1] = (hi_void*)((uintptr_t)pBuffer + widths * heights);
    inputFrame.v_frame.width = widths;
    inputFrame.v_frame.height = heights;
    inputFrame.v_frame.width_stride[0] = (hi_u32)widths;
    inputFrame.v_frame.width_stride[1] = (hi_u32)widths;
    inputFrame.v_frame.pixel_format = HI_PIXEL_FORMAT_YUV_SEMIPLANAR_420;
    inputFrame.v_frame.field = HI_VIDEO_FIELD_FRAME;  // Frame Or Field Mode

    hi_venc_jpeg_param stParamJpeg{};
    hi_mpi_venc_get_jpeg_param(g_chn_id, &stParamJpeg);
    stParamJpeg.qfactor = 100; // assign qfactor as 100
    hi_mpi_venc_set_jpeg_param(g_chn_id, &stParamJpeg);

    // Do encode
    hi_venc_stream stream{};
    hi_mpi_venc_send_frame(g_chn_id, &inputFrame, 10000); // time out 10000us
    stream.pack_cnt = 1;
    stream.pack = (hi_venc_pack*)malloc(sizeof(hi_venc_pack));
    hi_mpi_venc_get_stream(g_chn_id, &stream, -1);

    // Get output to host
    uint32_t dataLen = stream.pack[0].len - stream.pack[0].offset;
    std::vector<unsigned char> oBuf(dataLen);
    aclrtMemcpy(oBuf.data(), dataLen, stream.pack[0].addr + stream.pack[0].offset, dataLen, ACL_MEMCPY_DEVICE_TO_HOST);

    // save pic
    std::cout << "Writing JPEG file: " << outfile_path << ", dataLen: " << dataLen << std::endl;
    std::ofstream outputFile(outfile_path, std::ios::out | std::ios::binary);
    outputFile.write(reinterpret_cast<const char *>(oBuf.data()), static_cast<int>(dataLen));

    hi_mpi_venc_release_stream(g_chn_id, &stream);
    hi_mpi_venc_stop_chn(g_chn_id);
    hi_mpi_venc_destroy_chn(g_chn_id);
    hi_mpi_dvpp_free(pBuffer);
    free(stream.pack);

    aclrtDestroyContext(g_context);
    aclrtResetDevice(g_deviceId);
    aclFinalize();
}
