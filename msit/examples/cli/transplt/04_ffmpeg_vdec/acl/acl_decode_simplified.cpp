/**
 *  Copyright [2021] Huawei Technologies Co., Ltd
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
#include <getopt.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <signal.h>
#include <vector>
#include <string>
#include <sys/prctl.h>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <sstream>


#include "acl.h"
#include "acl_rt.h"
#include "hi_dvpp.h"

static uint32_t g_in_width = 3840; // Input stream width
static uint32_t g_in_height = 2160; // Input stream height

static uint32_t g_ref_frame_num = 8; // Number of reference frames [0, 16]
static uint32_t g_display_frame_num = 2; // Number of display frames [0, 16]
static uint32_t g_start_chn_num = 0; // Video decoder channel start number

static uint8_t* g_frame_addr[9999]; // Frame address
static uint64_t g_frame_len[9999]; // Frame size

static aclrtContext g_context = NULL;

static void pgmSave(unsigned char* yuv, uint32_t width, uint32_t height, std::string saveFileName)
{
    FILE* fp = fopen(saveFileName.c_str(), "wb");
    if (fp == nullptr) {
        printf("[%s][%d] Can't Open File %s \n", __FUNCTION__, __LINE__, saveFileName.c_str());
        return;
    }

    int ret = fprintf(fp, "P5\n%d %d\n%d\n", width, height, 255);
    if (ret < 0) {
        printf("[%s][%d] fprintf to file %s failed \n", __FUNCTION__, __LINE__, saveFileName.c_str());
        return;
    }
    ret = fwrite(yuv, 1, width * height, fp);
    if (ret < 0) {
        printf("[%s][%d] fwrite to file %s failed \n", __FUNCTION__, __LINE__, saveFileName.c_str());
        return;
    }
    ret = fclose(fp);
    if (ret < 0) {
        printf("[%s][%d] fclose file %s failed \n", __FUNCTION__, __LINE__, saveFileName.c_str());
        return;
    }
}

// convert YUV data to pgm data and write to a file
static void saveToPgmFile(std::string saveFileName, hi_video_frame frame, uint32_t chanId)
{
    uint8_t* addr = (uint8_t*)frame.virt_addr[0];
    uint32_t imageSize = frame.width * frame.height;
    int32_t ret = HI_SUCCESS;
    uint8_t* outImageBuf = nullptr;
    uint32_t outWidthStride = frame.width_stride[0];
    uint32_t outHeightStride = frame.height_stride[0];

    // malloc host memory
    ret = aclrtMallocHost((void **)&outImageBuf, imageSize);
    if (ret != ACL_SUCCESS) {
        printf("[%s][%d] malloc host memory %u failed, error code = %d.\n", __FUNCTION__, __LINE__, imageSize, ret);
        return;
    }

    if (outImageBuf == NULL) {
        return;
    }
    // Copy valid Y data to outImageBuf
    for (uint32_t i = 0; i < frame.height; i++) {
        ret = aclrtMemcpy(outImageBuf + i * frame.width, frame.width, addr + i * outWidthStride,
            frame.width, ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) {
            printf("[%s][%d] Copy aclrtMemcpy %u from device to host failed, error code = %d.\n",
                __FUNCTION__, __LINE__, imageSize, ret);
            aclrtFreeHost(outImageBuf);
            return;
        }
    }

    pgmSave(outImageBuf, frame.width, frame.height, saveFileName);
    aclrtFreeHost(outImageBuf);
    return;
}

// Create video decoder channel, channel number is g_start_chn_num
static int32_t vdecCreate()
{
    hi_vdec_chn_attr chnAttr{};
    hi_data_bit_width bitWidth = HI_DATA_BIT_WIDTH_8;

    chnAttr.type = HI_PT_H264; // Input stream is H264
    chnAttr.mode = HI_VDEC_SEND_MODE_FRAME; // Only support frame mode
    chnAttr.pic_width = g_in_width;
    chnAttr.pic_height = g_in_height;
    chnAttr.stream_buf_size = g_in_width * g_in_height * 3 / 2;
    chnAttr.frame_buf_cnt = g_ref_frame_num + g_display_frame_num + 1;

    hi_pic_buf_attr buf_attr{g_in_width, g_in_height, 0,
                                bitWidth, HI_PIXEL_FORMAT_YUV_SEMIPLANAR_420, HI_COMPRESS_MODE_NONE};
    chnAttr.frame_buf_size = hi_vdec_get_pic_buf_size(chnAttr.type, &buf_attr);

    // Configure video decoder channel attribute
    chnAttr.video_attr.ref_frame_num = g_ref_frame_num;
    chnAttr.video_attr.temporal_mvp_en = HI_TRUE;
    chnAttr.video_attr.tmv_buf_size = hi_vdec_get_tmv_buf_size(chnAttr.type, g_in_width, g_in_height);

    hi_mpi_vdec_create_chn(g_start_chn_num, &chnAttr);

    hi_vdec_chn_param chnParam;
    // Get channel parameter
    hi_mpi_vdec_get_chn_param(g_start_chn_num, &chnParam);
    chnParam.video_param.dec_mode = HI_VIDEO_DEC_MODE_IPB;
    chnParam.video_param.compress_mode = HI_COMPRESS_MODE_HFBC;
    chnParam.video_param.video_format = HI_VIDEO_FORMAT_TILE_64x16;
    chnParam.display_frame_num = g_display_frame_num;
    chnParam.video_param.out_order = HI_VIDEO_OUT_ORDER_DISPLAY; // Display sequence

    // Set channel parameter
    hi_mpi_vdec_set_chn_param(g_start_chn_num, &chnParam);

    // Decoder channel start receive stream
    hi_mpi_vdec_start_recv_stream(g_start_chn_num);
    return HI_SUCCESS;
}

// Cutting stream to frame
static void getEveryFrame(int32_t chanId, uint8_t* const inputFileBuf, uint32_t* const frameCount, uint32_t fileSize,
    hi_payload_type type, uint8_t* dataDev)
{
    int32_t i = 0;
    int32_t usedBytes = 0;
    uint32_t count = 0;
    int32_t readLen = fileSize - usedBytes;
    uint8_t* bufPointer = inputFileBuf + usedBytes;
    bool isFindStart = false;
    bool isFindEnd = false;

    while (readLen > 0) {
        isFindStart = false;
        isFindEnd = false;

        // H264
        for (i = 0; i < readLen - 8; i++) {
            int32_t tmp = bufPointer[i + 3] & 0x1F;
            // Find 00 00 01
            if ((bufPointer[i] == 0) && (bufPointer[i + 1] == 0) && (bufPointer[i + 2] == 1) &&
                (((tmp == 0x5 || tmp == 0x1) && ((bufPointer[i + 4] & 0x80) == 0x80)) ||
                (tmp == 20 && (bufPointer[i + 7] & 0x80) == 0x80))) {
                isFindStart = true;
                i += 8;
                break;
            }
        }

        for (; i < readLen - 8; i++) {
            int32_t tmp = bufPointer[i + 3] & 0x1F;
            // Find 00 00 01
            if ((bufPointer[i] == 0) && (bufPointer[i + 1] == 0) && (bufPointer[i + 2] == 1) &&
                ((tmp == 15) || (tmp == 7) || (tmp == 8) || (tmp == 6) ||
                ((tmp == 5 || tmp == 1) && ((bufPointer[i + 4] & 0x80) == 0x80)) ||
                (tmp == 20 && (bufPointer[i + 7] & 0x80) == 0x80))) {
                isFindEnd = true;
                break;
            }
        }

        if (i > 0) {
            readLen = i;
        }

        if (!isFindStart) {
            printf("can not find H264 start code! readLen %d, usedBytes %d.!\n", readLen, usedBytes);
        }
        if (!isFindEnd) {
            readLen = i + 8;
        }

        g_frame_addr[count] = (bufPointer - inputFileBuf) + dataDev; // Record frame address
        g_frame_len[count] = readLen; // Record frame size
        count++;
        usedBytes = usedBytes + readLen;

        bufPointer = inputFileBuf + usedBytes;
        readLen = fileSize - usedBytes;
    }
    // Frame count
    *frameCount = count;
}

static int32_t decode(std::string input_file_name, std::string output_file_name)
{
    uint32_t chanId = 0;

    aclrtSetCurrentContext(g_context);

    std::ifstream file(input_file_name.c_str(), std::ios::binary | std::ios::ate);
    std::streamsize fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> inputFileBuf(fileSize);
    file.read(inputFileBuf.data(), fileSize);
    file.close();

    uint8_t* dataDev = HI_NULL;
    int32_t ret = HI_SUCCESS;

    // alloc device inbuffer mem
    ret = hi_mpi_dvpp_malloc(0, (void **)&dataDev, fileSize);
    if (ret != 0) {
        printf("[%s][%d] Malloc device memory failed.\n", __FUNCTION__, __LINE__);
        return HI_FAILURE;
    }

    // copy host to device
    ret = aclrtMemcpy(dataDev, fileSize, inputFileBuf.data(), fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        hi_mpi_dvpp_free(dataDev);
        printf("[%s][%d] Copy host memcpy to device failed, error code = %d.\n", __FUNCTION__, __LINE__, ret);
        return HI_FAILURE;
    }

    uint32_t frameCount = 0;
    hi_payload_type type = HI_PT_H264;
    // Cutting stream
    getEveryFrame(chanId, (uint8_t*)inputFileBuf.data(), &frameCount, fileSize, type, dataDev);

    void* sendStreamBuffer = nullptr;
    uint32_t sendStreamBufferSize = g_in_width * g_in_height * 3 / 2;

    ret = hi_mpi_dvpp_malloc(0, &sendStreamBuffer, sendStreamBufferSize);
    if (ret != HI_SUCCESS) {
        hi_mpi_dvpp_free(dataDev);
        printf("[%s][%d] hi_mpi_dvpp_malloc failed.\n", __FUNCTION__, __LINE__);
        return HI_FAILURE;
    }

    // Delay one second
    usleep(1000000);

    // Start send stream
    hi_vdec_stream stream;
    hi_vdec_pic_info outPicInfo;
    uint32_t decodeCount = 0;
    int32_t timeOut = 1000;

    hi_vdec_stream receive_stream;
    hi_video_frame_info frame;
    int32_t decResult = 0; // Decode result
    void* outputBuffer = nullptr;
    int32_t getStreamFailCnt = 0;
    int32_t successCnt = 0;
    int32_t failCnt = 0;
    hi_vdec_supplement_info stSupplement{};

    while (decodeCount < frameCount) {
        stream.addr = g_frame_addr[decodeCount]; // Configure input stream address
        stream.len = g_frame_len[decodeCount]; // Configure input stream size
        stream.end_of_frame = HI_TRUE; // Configure flag of frame end
        stream.end_of_stream = HI_FALSE; // Configure flag of stream end

        outPicInfo.width = 0; // Output image width, supports resize, set 0 means no resize
        outPicInfo.height = 0; // Output image height, supports resize, set 0 means no resize
        outPicInfo.width_stride = g_in_width; // Output memory width stride
        outPicInfo.height_stride = g_in_height; // Output memory height stride
        outPicInfo.pixel_format = HI_PIXEL_FORMAT_YUV_SEMIPLANAR_420; // Configure output format

        stream.need_display = HI_TRUE;
        outPicInfo.vir_addr = (uint64_t)sendStreamBuffer;
        outPicInfo.buffer_size = sendStreamBufferSize;

        ret = hi_mpi_vdec_send_stream(chanId, &stream, &outPicInfo, timeOut);
        if (ret != HI_SUCCESS) {
            printf("[%s][%d] hi_mpi_vdec_send_stream Fail, Error Code = %x \n", __FUNCTION__, __LINE__, ret);
            break;
        }

        usleep(1000);

        ret = hi_mpi_vdec_get_frame(chanId, &frame, &stSupplement, &receive_stream, timeOut);
        if (ret != HI_SUCCESS) {
            if (getStreamFailCnt++ < 1000) {
                printf("[%s][%d] hi_mpi_vdec_get_frame Fail[%d], Error Code = %x, retrying... \n",
                       __FUNCTION__, __LINE__, getStreamFailCnt, ret);
                continue;
            } else {
                break;
            }
        }

        decodeCount++;
        outputBuffer = (void*)frame.v_frame.virt_addr[0];
        decResult = frame.v_frame.frame_flag;
        if (decResult == 0) {
            successCnt++;
            printf("[%s][%d] GetFrame Success, Decode Success[%d] \n", __FUNCTION__, __LINE__, successCnt);
        } else  {
            failCnt++;
            printf("[%s][%d] GetFrame Success, Decode Fail[%d] \n", __FUNCTION__, __LINE__, failCnt);
        }

        // Decode result write to a file
        if ((decResult == 0) && (outputBuffer != NULL) && (receive_stream.need_display == HI_TRUE)) {
            static int32_t writeFileCnt = 1;
            std::ostringstream sstream;
            sstream << output_file_name.c_str() << "-" << writeFileCnt << ".pgm";
            std::string saveFileName = sstream.str();
            saveToPgmFile(saveFileName, frame.v_frame, chanId);
            writeFileCnt++;
        }
        hi_mpi_vdec_release_frame(chanId, &frame);
    }

    hi_mpi_dvpp_free(dataDev);
    return HI_SUCCESS;
}

static int32_t hiDvppInit()
{
    aclInit(NULL);
    aclrtSetDevice(0);
    aclrtCreateContext(&g_context, 0);
    hi_mpi_sys_init();
    return HI_SUCCESS;
}

static void hiDvppDeinit()
{
    hi_mpi_vdec_stop_recv_stream(g_start_chn_num);
    hi_mpi_vdec_destroy_chn(g_start_chn_num);
    hi_mpi_sys_exit();
    aclrtDestroyContext(g_context);
    aclrtResetDevice(0);
    aclFinalize();
}

int32_t main(int32_t argc, char *argv[])
{
    printf("[%s][%d] Video decode started.\n", __FUNCTION__, __LINE__);
    int32_t ret = HI_SUCCESS;

    if (argc <= 2) {
        printf("Usage: %s <input file> <output file>\n", argv[0]);
        return HI_FAILURE;
    }
    std::string input_file_name = std::string(argv[1]);
    std::string output_file_name = std::string(argv[2]);

    hiDvppInit();
    vdecCreate();
    decode(input_file_name, output_file_name);
    hiDvppDeinit();

    printf("[%s][%d] Video decode finished.\n", __FUNCTION__, __LINE__);
    return 0;
}
