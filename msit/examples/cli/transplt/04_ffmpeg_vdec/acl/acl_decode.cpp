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
#include <sstream>

#include "acl.h"
#include "acl_rt.h"
#include "hi_dvpp.h"

static std::string g_inputFileName = std::string("infile"); // Input stream file name
static std::string g_outputFileName = std::string("outfile"); // Output file name
static uint32_t g_in_width = 3840; // Input stream width
static uint32_t g_in_height = 2160; // Input stream height
static uint32_t g_in_bitwidth = 8; // Input stream bit width, 8 or 10

static uint32_t g_ref_frame_num = 8; // Number of reference frames [0, 16]
static uint32_t g_display_frame_num = 2; // Number of display frames [0, 16]
static uint32_t g_alloc_num = 20; // Number of out buffer
static uint32_t g_start_chn_num = 0; // Video decoder channel start number

static std::vector<void*> g_outBufferPool; // Out buffer pool
static pthread_mutex_t g_outBufferPoolLock; // Lock of out buffer pool

static const uint32_t g_GB = 1024 * 1024 * 1024;

static uint32_t g_chan_create_state = 0; // Video decoder channel state, 0: not created, 1: created
static uint32_t g_exit = 0; // Force Exit Flag
static uint32_t g_send_exit_state = 0; // State of send thread
static uint32_t g_get_exit_state = 0; // State of get thread
static uint32_t g_send_thread_id = 0;
static uint32_t g_get_thread_id = 0;
static pthread_t g_vdecSendThread = 0;
static pthread_t g_vdecGetThread = 0;

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
        printf("[%s][%d] Chn %u malloc host memory %u failed, error code = %d.\n",
            __FUNCTION__, __LINE__, chanId, imageSize, ret);
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
            printf("[%s][%d] Chn %u Copy aclrtMemcpy %u from device to host failed, error code = %d.\n",
                __FUNCTION__, __LINE__, chanId, imageSize, ret);
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
    int32_t ret = HI_SUCCESS;
    hi_vdec_chn_attr chnAttr{};
    hi_data_bit_width bitWidth = HI_DATA_BIT_WIDTH_8;

    chnAttr.type = HI_PT_H264; // Input stream is H264
    // Configure channel attribute
    chnAttr.mode = HI_VDEC_SEND_MODE_FRAME; // Only support frame mode
    chnAttr.pic_width = g_in_width;
    chnAttr.pic_height = g_in_height;
    // Stream buffer size, Recommended value is width * height * 3 / 2
    chnAttr.stream_buf_size = g_in_width * g_in_height * 3 / 2;
    chnAttr.frame_buf_cnt = g_ref_frame_num + g_display_frame_num + 1;

    hi_pic_buf_attr buf_attr{g_in_width, g_in_height, 0,
                                bitWidth, HI_PIXEL_FORMAT_YUV_SEMIPLANAR_420, HI_COMPRESS_MODE_NONE};
    chnAttr.frame_buf_size = hi_vdec_get_pic_buf_size(chnAttr.type, &buf_attr);

    // Configure video decoder channel attribute
    chnAttr.video_attr.ref_frame_num = g_ref_frame_num;
    chnAttr.video_attr.temporal_mvp_en = HI_TRUE;
    chnAttr.video_attr.tmv_buf_size = hi_vdec_get_tmv_buf_size(chnAttr.type, g_in_width, g_in_height);

    ret = hi_mpi_vdec_create_chn(g_start_chn_num, &chnAttr);
    if (ret != HI_SUCCESS) {
        printf("[%s][%d] Chn %u, hi_mpi_vdec_create_chn Fail, ret = %x \n",
               __FUNCTION__, __LINE__, g_start_chn_num, ret);
        return ret;
    }
    g_chan_create_state = 1;

    hi_vdec_chn_param chnParam;
    // Get channel parameter
    ret = hi_mpi_vdec_get_chn_param(g_start_chn_num, &chnParam);
    if (ret != HI_SUCCESS) {
        printf("[%s][%d] Chn %u, hi_mpi_vdec_get_chn_param Fail, ret = %x \n",
               __FUNCTION__, __LINE__, g_start_chn_num, ret);
        return ret;
    }
    chnParam.video_param.dec_mode = HI_VIDEO_DEC_MODE_IPB;
    chnParam.video_param.compress_mode = HI_COMPRESS_MODE_HFBC;
    chnParam.video_param.video_format = HI_VIDEO_FORMAT_TILE_64x16;
    chnParam.display_frame_num = g_display_frame_num;
    chnParam.video_param.out_order = HI_VIDEO_OUT_ORDER_DISPLAY; // Display sequence

    // Set channel parameter
    ret = hi_mpi_vdec_set_chn_param(g_start_chn_num, &chnParam);
    if (ret != HI_SUCCESS) {
        printf("[%s][%d] Chn %u, hi_mpi_vdec_set_chn_param Fail, ret = %x \n",
               __FUNCTION__, __LINE__, g_start_chn_num, ret);
        return ret;
    }

    // Decoder channel start receive stream
    ret = hi_mpi_vdec_start_recv_stream(g_start_chn_num);
    if (ret != HI_SUCCESS) {
        printf("[%s][%d] Chn %u, hi_mpi_vdec_start_recv_stream Fail, ret = %x \n",
               __FUNCTION__, __LINE__, g_start_chn_num, ret);
        return ret;
    }
    return ret;
}

static void releaseOutbuffer()
{
    void* outBuffer = nullptr;
    while (g_outBufferPool.empty() == false) {
        outBuffer = g_outBufferPool.back();
        g_outBufferPool.pop_back();
        hi_mpi_dvpp_free(outBuffer);
    }
}

static void vdecResetChn(uint32_t chanId)
{
    int32_t ret = HI_SUCCESS;
    hi_vdec_chn_status status{};

    ret = hi_mpi_vdec_stop_recv_stream(chanId);
    if (ret != HI_SUCCESS) {
        printf("[%s][%d] Chn %u, hi_mpi_vdec_stop_recv_stream Fail, ret = %x \n", __FUNCTION__, __LINE__, chanId, ret);
        return;
    }
    // reset channel
    ret = hi_mpi_vdec_reset_chn(chanId);
    if (ret != HI_SUCCESS) {
        printf("[%s][%d] Chn %u, hi_mpi_vdec_reset_chn Fail, ret = %x \n", __FUNCTION__, __LINE__, chanId, ret);
        return;
    }

    ret = hi_mpi_vdec_start_recv_stream(chanId);
    if (ret != HI_SUCCESS) {
        printf("[%s][%d] Chn %u, hi_mpi_vdec_start_recv_stream Fail, ret = %x \n",
               __FUNCTION__, __LINE__, chanId, ret);
        return;
    }
    printf("[%s][%d] Chn %u, reset chn success \n", __FUNCTION__, __LINE__, chanId);
    return;
}

static void waitVdecEnd()
{
    int32_t ret = HI_SUCCESS;
    int32_t waitTimes;
    int32_t sleepTime = 10000; // 10000us
    hi_vdec_chn_status status{};
    hi_vdec_chn_status pre_status{};

    if (g_vdecSendThread != 0) {
        // Wait send thread exit
        ret = pthread_join(g_vdecSendThread, nullptr);
    }
    g_vdecSendThread = 0;

    waitTimes = 0;
    // Wait channel decode over
    while (g_exit == 0) {
        ret = hi_mpi_vdec_query_status(g_start_chn_num, &status);
        if (ret != HI_SUCCESS) {
            printf("[%s][%d] Chn %u, hi_mpi_vdec_query_status Fail, ret = %x \n",
                   __FUNCTION__, __LINE__, g_start_chn_num, ret);
            break;
        }
        if (((status.left_stream_bytes == 0) && (status.left_decoded_frames == 0)) || (g_get_exit_state == 1)) {
            break;
        }
        if (status.left_decoded_frames == pre_status.left_decoded_frames) {
            waitTimes += sleepTime;
        } else {
            waitTimes = 0;
        }
        pre_status = status;
        // 10000us
        usleep(sleepTime);

        if (waitTimes >= 5000000) { // 5000000 us
            vdecResetChn(g_start_chn_num);
            break;
        }
    }
    // 1000000us
    usleep(1000000);

    // Notify get thread exit
    g_get_exit_state = 1;

    if (g_vdecGetThread != 0) {
        // Wait get thread exit
        ret = pthread_join(g_vdecGetThread, nullptr);
    }
    g_vdecGetThread = 0;
}

static void vdecDestroy()
{
    int32_t ret = HI_SUCCESS;
    if (g_chan_create_state == 1) {
        // Decoder channel stop receive stream
        ret = hi_mpi_vdec_stop_recv_stream(g_start_chn_num);
        if (ret != HI_SUCCESS) {
            printf("[%s][%d] Chn %u, hi_mpi_vdec_stop_recv_stream Fail, ret = %x \n",
                   __FUNCTION__, __LINE__, g_start_chn_num, ret);
        }
        // Destroy channel
        ret = hi_mpi_vdec_destroy_chn(g_start_chn_num);
        if (ret != HI_SUCCESS) {
            printf("[%s][%d] Chn %u, hi_mpi_vdec_destroy_chn Fail, ret = %x \n",
                   __FUNCTION__, __LINE__, g_start_chn_num, ret);
        }
        releaseOutbuffer();
    }
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
            printf("Chn %d can not find H264 start code!readLen %d, usedBytes %d.!\n", chanId, readLen, usedBytes);
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

static void* sendStream(void* const chanNum)
{
    prctl(PR_SET_NAME, "VdecSendStream", 0, 0, 0);
    uint32_t chanId = *(uint32_t*)chanNum;

    aclError aclRet = aclrtSetCurrentContext(g_context);
    if (aclRet != ACL_SUCCESS) {
        printf("[%s][%d] Chn %u set current context failed, error code = %d", __FUNCTION__, __LINE__, chanId, aclRet);
        return (void*)(HI_FAILURE);
    }

    // Open input stream file
    FILE* fpInputFile = nullptr;
    fpInputFile = fopen(g_inputFileName.c_str(), "rb");
    if (fpInputFile == nullptr) {
        printf("[%s][%d] Chn %u Can't open file %s \n", __FUNCTION__, __LINE__, chanId, g_inputFileName.c_str());
        return (void*)(HI_FAILURE);
    }

    // Calculate input stream file size
    uint32_t fileSize = 0;
    int ret = fseek(fpInputFile, 0L, SEEK_END);
    if (ret != 0) {
        printf("[%s][%d] Chn %u fseek file %s failed \n", __FUNCTION__, __LINE__, chanId, g_inputFileName.c_str());
        return (void*)(HI_FAILURE);
    }
    fileSize = ftell(fpInputFile);
    ret = fseek(fpInputFile, 0L, SEEK_SET);
    if (ret != 0) {
        printf("[%s][%d] Chn %u fseek file %s failed \n", __FUNCTION__, __LINE__, chanId, g_inputFileName.c_str());
        return (void*)(HI_FAILURE);
    }

    if (fileSize == 0 || fileSize > g_GB) {
        printf("[%s][%d] Chn %u InputFile size is invalid %d Fail \n", __FUNCTION__, __LINE__, chanId, fileSize);
        return (void*)(HI_FAILURE);
    }

    // Alloc buffer for all input stream file
    uint8_t* inputFileBuf = NULL;
    inputFileBuf = (uint8_t*)malloc(fileSize);
    if (inputFileBuf == NULL) {
        fclose(fpInputFile);
        printf("[%s][%d] Chn %u Malloc InputFile Buffer Fail \n", __FUNCTION__, __LINE__, chanId);
        return (void*)(HI_FAILURE);
    }

    // Read input stream file
    uint32_t readLen = 0;
    readLen = fread(inputFileBuf, 1, fileSize, fpInputFile);
    if (readLen != fileSize) {
        fclose(fpInputFile);
        free(inputFileBuf);
        printf("[%s][%d] Chn %u Read InputFile Fail \n", __FUNCTION__, __LINE__, chanId);
        return (void*)(HI_FAILURE);
    }

    uint8_t* dataDev = HI_NULL;
    ret = HI_SUCCESS;

    // alloc device inbuffer mem
    ret = hi_mpi_dvpp_malloc(0, (void **)&dataDev, fileSize);
    if (ret != 0) {
        fclose(fpInputFile);
        free(inputFileBuf);
        printf("[%s][%d] Chn %u Malloc device memory %u failed.\n", __FUNCTION__, __LINE__, chanId, fileSize);
        return (hi_void *)(HI_FAILURE);
    }

    // copy host to device
    ret = aclrtMemcpy(dataDev, fileSize, inputFileBuf, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        fclose(fpInputFile);
        free(inputFileBuf);
        hi_mpi_dvpp_free(dataDev);
        printf("[%s][%d] Chn %u Copy host memcpy to device failed, error code = %d.\n",
               __FUNCTION__, __LINE__, chanId, ret);
        return (hi_void *)(HI_FAILURE);
    }

    uint32_t frameCount = 0;
    hi_payload_type type = HI_PT_H264;
    // Cutting stream
    getEveryFrame(chanId, inputFileBuf, &frameCount, fileSize, type, dataDev);
    if (frameCount == 0) {
        return (hi_void *)(HI_FAILURE);
    }

    // Using out buffer pool in order to prevent system oom
    void* outBuffer = nullptr;
    uint32_t outBufferSize = 0;
    // Out format is YUV420sp
    outBufferSize = g_in_width * g_in_height * 3 / 2;

    // Alloc out buffer
    for (uint32_t i = 0; i < g_alloc_num; i++) {
        ret = hi_mpi_dvpp_malloc(0, &outBuffer, outBufferSize); // Alloc Vdec out buffer must use hi_mpi_dvpp_malloc
        if (ret != HI_SUCCESS) {
            fclose(fpInputFile);
            free(inputFileBuf);
            hi_mpi_dvpp_free(dataDev);
            printf("[%s][%d] Chn %u hi_mpi_dvpp_malloc failed.\n", __FUNCTION__, __LINE__, chanId);
            return (void*)(HI_FAILURE);
        }
        // Put buffer to pool
        g_outBufferPool.push_back(outBuffer);
    }

    // Delay one second
    usleep(1000000);

    // Start send stream
    hi_vdec_stream stream;
    hi_vdec_pic_info outPicInfo;
    uint32_t readCount = 0;
    hi_pixel_format outFormat = HI_PIXEL_FORMAT_YUV_SEMIPLANAR_420;
    uint32_t circleTimes = 0;
    uint32_t sendOneFrameCnt = 0;
    int32_t timeOut = 1000;
    while (g_send_exit_state == 0) {
        stream.addr = g_frame_addr[readCount]; // Configure input stream address
        stream.len = g_frame_len[readCount]; // Configure input stream size
        stream.end_of_frame = HI_TRUE; // Configure flage of frame end
        stream.end_of_stream = HI_FALSE; // Configure flage of stream end

        outPicInfo.width = 0; // Output image width, supports resize, set 0 means no resize
        outPicInfo.height = 0; // Output image height, supports resize, set 0 means no resize
        outPicInfo.width_stride = g_in_width; // Output memory width stride
        outPicInfo.height_stride = g_in_height; // Output memory height stride
        outPicInfo.pixel_format = outFormat; // Configure output format

        int32_t mallocCount = 0;
        int32_t tryTimes = 20000;
        // Get out buffer from pool
        while (g_send_exit_state == 0) {
            mallocCount++;
            (void)pthread_mutex_lock(&g_outBufferPoolLock);
            if (g_outBufferPool.empty() == false) {
                outBuffer = g_outBufferPool.back();
                g_outBufferPool.pop_back();
                (void)pthread_mutex_unlock(&g_outBufferPoolLock);
                break;
            } else {
                (void)pthread_mutex_unlock(&g_outBufferPoolLock);
                usleep(1000); // 1000us
            }

            if (mallocCount >= tryTimes) {
                printf("[%s][%d] Chn %u DvppMalloc From Pool Fail, Try again\n", __FUNCTION__, __LINE__, chanId);
                mallocCount = 0;
            }
        }

        stream.need_display = HI_TRUE;
        outPicInfo.vir_addr = (uint64_t)outBuffer;
        outPicInfo.buffer_size = outBufferSize;

        readCount = (readCount + 1) % frameCount;
        // Finish sending stream one times
        if (readCount == 0) {
            circleTimes++;
        }

        do {
            sendOneFrameCnt++;
            // Send one frame data
            ret = hi_mpi_vdec_send_stream(chanId, &stream, &outPicInfo, timeOut);
            if (sendOneFrameCnt > 30) { // if send stream timeout 30 times, end the decode process
                vdecResetChn(chanId);
                sendOneFrameCnt = 0;
                break;
            }
        } while (ret == HI_ERR_VDEC_BUF_FULL); // Try again

        if (ret != HI_SUCCESS) {
            printf("[%s][%d] Chn %u hi_mpi_vdec_send_stream Fail, Error Code = %x \n",
                   __FUNCTION__, __LINE__, chanId, ret);
            break;
        }

        sendOneFrameCnt = 0;
        if (circleTimes >= 1) {
            break;
        }
    }

    // Send stream end flage
    stream.addr = NULL;
    stream.len = 0;
    stream.end_of_frame = HI_FALSE;
    stream.end_of_stream = HI_TRUE; // Stream end flage
    outPicInfo.vir_addr = 0;
    outPicInfo.buffer_size = 0;
    hi_mpi_vdec_send_stream(chanId, &stream, &outPicInfo, -1);

    ret = fclose(fpInputFile);
    if (ret != 0) {
        printf("[%s][%d] fclose file %s failed \n", __FUNCTION__, __LINE__, g_inputFileName.c_str());
        return (void*)(HI_FAILURE);
    }
    free(inputFileBuf);
    hi_mpi_dvpp_free(dataDev);
    printf("[%s][%d] Chn %u sendStream Thread Exit \n", __FUNCTION__, __LINE__, chanId);
    return (hi_void *)HI_SUCCESS;
}

static void* getPic(void* const chanNum)
{
    prctl(PR_SET_NAME, "VdecGetPic", 0, 0, 0);
    uint32_t chanId = *(uint32_t*)chanNum;

    aclError aclRet = aclrtSetCurrentContext(g_context);
    if (aclRet != ACL_SUCCESS) {
        printf("[%s][%d] Chn %u set current context failed, error code = %d",
               __FUNCTION__, __LINE__, chanId, aclRet);
        g_get_exit_state = 1;
        return (void*)(HI_FAILURE);
    }

    int32_t ret = HI_SUCCESS;
    hi_video_frame_info frame;
    hi_vdec_stream stream;
    int32_t decResult = 0; // Decode result
    void* outputBuffer = nullptr;
    int32_t successCnt = 0;
    int32_t failCnt = 0;
    int32_t timeOut = 1000;
    int32_t writeFileCnt = 1;
    hi_vdec_supplement_info stSupplement{};

    while (g_get_exit_state == 0) {
        ret = hi_mpi_vdec_get_frame(chanId, &frame, &stSupplement, &stream, timeOut);
        if (ret != HI_SUCCESS) {
            // 500us
            usleep(500);
            continue;
        }

        outputBuffer = (void*)frame.v_frame.virt_addr[0];
        decResult = frame.v_frame.frame_flag;
        if (decResult == 0) { // 0: Decode success
            successCnt++;
            printf("[%s][%d] Chn %u GetFrame Success, Decode Success[%d] \n",
                   __FUNCTION__, __LINE__, chanId, successCnt);
        } else  {
            failCnt++;
            printf("[%s][%d] Chn %u GetFrame Success, Decode Fail[%d] \n", __FUNCTION__, __LINE__, chanId, failCnt);
        }

        // Decode result write to a file
        if ((decResult == 0) && (outputBuffer != NULL) && (stream.need_display == HI_TRUE)) {
            FILE* fp = nullptr;

            // Configure file name
            std::ostringstream sstream;
            sstream << g_outputFileName.c_str() << "-" << writeFileCnt << ".pgm";
            std::string saveFileName = sstream.str();

            saveToPgmFile(saveFileName, frame.v_frame, chanId);
            writeFileCnt++;
        }
        if (outputBuffer != nullptr) {
            // Put out buffer to pool
            (void)pthread_mutex_lock(&g_outBufferPoolLock);
            g_outBufferPool.push_back(outputBuffer);
            (void)pthread_mutex_unlock(&g_outBufferPoolLock);
        }
        // Release Frame
        ret = hi_mpi_vdec_release_frame(chanId, &frame);

        if (ret != HI_SUCCESS) {
            printf("[%s][%d] Chn %u hi_mpi_vdec_release_frame Fail, Error Code = %x \n",
                   __FUNCTION__, __LINE__, chanId, ret);
        }
    }
    printf("[%s][%d] Chn %u getPic Thread Exit \n", __FUNCTION__, __LINE__, chanId);
    return (void*)HI_SUCCESS;
}

static int32_t createSendStreamThread()
{
    int32_t ret;
    g_vdecSendThread = 0;
    g_send_thread_id = g_start_chn_num;
    // Create send thread
    ret = pthread_create(&g_vdecSendThread, 0, sendStream, (void*)&g_send_thread_id);
    if (ret != 0) {
        printf("[%s][%d] Chn %u, create send stream thread Fail, ret = %d \n",
               __FUNCTION__, __LINE__, g_start_chn_num, ret);
        g_vdecSendThread = 0;
        return ret;
    }

    return ret;
}

static int32_t createGetPicThread()
{
    int32_t ret;
    g_vdecGetThread = 0;
    // Create get thread
    ret = pthread_create(&g_vdecGetThread, 0, getPic, (void*)&g_get_thread_id);
    if (ret != 0) {
        printf("[%s][%d] Chn %u, create get pic thread Fail, ret = %d \n",
               __FUNCTION__, __LINE__, g_start_chn_num, ret);
        g_vdecGetThread = 0;
        return ret;
    }

    return ret;
}

static void stopSendStreamThread()
{
    g_send_exit_state = 1;
    g_exit = 1;
}

static void stopGetPicThread()
{
    g_get_exit_state = 1;
    g_send_exit_state = 1;
    g_exit = 1;
}

// Parse input parameters
static int32_t getOption(int32_t argc, char **argv)
{
    if (argc <= 2) {
        printf("Usage: %s <input file> <output file>\n", argv[0]);
        return HI_FAILURE;
    }

    g_inputFileName = std::string(argv[1]);
    g_outputFileName = std::string(argv[2]);
    return HI_SUCCESS;
}

static int32_t hiDvppInit()
{
    aclError aclRet = aclInit(NULL);
    if (aclRet != ACL_SUCCESS) {
        printf("[%s][%d] aclInit failed, error code = %d.\n", __FUNCTION__, __LINE__, aclRet);
        return HI_FAILURE;
    }
    printf("[%s][%d] aclInit Success.\n", __FUNCTION__, __LINE__);

    aclRet = aclrtSetDevice(0);
    if (aclRet != ACL_SUCCESS) {
        printf("[%s][%d] aclrtSetDevice 0 failed, error code = %d.\n", __FUNCTION__, __LINE__, aclRet);
        aclFinalize();
        return HI_FAILURE;
    }
    printf("[%s][%d] aclrtSetDevice 0 Success.\n", __FUNCTION__, __LINE__);

    aclRet = aclrtCreateContext(&g_context, 0);
    if (aclRet != ACL_SUCCESS) {
        printf("[%s][%d] aclrtCreateContext failed, error code = %d.\n", __FUNCTION__, __LINE__, aclRet);
        aclrtResetDevice(0);
        aclFinalize();
        return HI_FAILURE;
    }
    printf("[%s][%d] aclrtCreateContext Success\n", __FUNCTION__, __LINE__);

    int32_t ret = hi_mpi_sys_init();
    if (ret != HI_SUCCESS) {
        printf("[%s][%d] hi_mpi_sys_init failed, error code = %x\n", __FUNCTION__, __LINE__, ret);
        aclrtDestroyContext(g_context);
        aclrtResetDevice(0);
        aclFinalize();
        return HI_FAILURE;
    }

    printf("[%s][%d] Dvpp system init success\n", __FUNCTION__, __LINE__);
    return HI_SUCCESS;
}

static void hiDvppDeinit()
{
    int32_t ret = hi_mpi_sys_exit();
    if (ret != HI_SUCCESS) {
        printf("[%s][%d] hi_mpi_sys_exit failed, error code = %x.\n", __FUNCTION__, __LINE__, ret);
    }

    aclError aclRet = aclrtDestroyContext(g_context);
    if (aclRet != ACL_SUCCESS) {
        printf("[%s][%d] aclrtDestroyContext failed, error code = %d.\n", __FUNCTION__, __LINE__, aclRet);
    }

    aclRet = aclrtResetDevice(0);
    if (aclRet != ACL_SUCCESS) {
        printf("[%s][%d] aclrtResetDevice 0 failed, error code = %d.\n", __FUNCTION__, __LINE__, aclRet);
    }

    aclRet = aclFinalize();
    if (aclRet != ACL_SUCCESS) {
        printf("[%s][%d] aclFinalize failed, error code = %d.\n", __FUNCTION__, __LINE__, aclRet);
    }

    printf("[%s][%d] Dvpp system exit success\n", __FUNCTION__, __LINE__);
}

int32_t main(int32_t argc, char *argv[])
{
    int32_t ret = HI_SUCCESS;

    // Parse input parameters
    ret = getOption(argc, &(*argv));
    if (ret != HI_SUCCESS) {
        printf("[%s][%d] getOption Fail \n", __FUNCTION__, __LINE__);
        return 0;
    }

    // Dvpp system init
    ret = hiDvppInit();
    if (ret != HI_SUCCESS) {
        printf("[%s][%d] hiDvppInit failed!\n", __FUNCTION__, __LINE__);
        return 0;
    }

    // Create video decoder
    ret = vdecCreate();
    if (ret != HI_SUCCESS) {
        printf("[%s][%d] VdecStart failed!\n", __FUNCTION__, __LINE__);
        // Destroy video decoder
        vdecDestroy();
        // Dvpp system exit
        hiDvppDeinit();
        return 0;
    }
    // Lock init
    pthread_mutex_init(&g_outBufferPoolLock, nullptr);

    // Create threads for sending stream
    ret = createSendStreamThread();
    if (ret != 0) {
        // If create thread fail, stop all send stream thread
        stopSendStreamThread();
    } else {
        // Create threads for getting result
        ret = createGetPicThread();
        if (ret != 0) {
            // If create thread fail, stop all get pic thread
            stopGetPicThread();
        }
    }

    // Wait decoding is complete.
    waitVdecEnd();
    // Destroy init
    pthread_mutex_destroy(&g_outBufferPoolLock);
    // Destroy video decoder
    vdecDestroy();
    // Dvpp system exit
    hiDvppDeinit();
    return 0;
}
