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

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sstream>

extern "C" {
#include <libavcodec/avcodec.h>
}

static const uint32_t INBUF_SIZE = 4096;

static void pgmSave(unsigned char *buf, int wrap, int xsize, int ysize, std::string filename)
{
    FILE *f;
    int i;

    f = fopen(filename.c_str(),"wb");
    int ret = fprintf(f, "P5\n%d %d\n%d\n", xsize, ysize, 255);
    if (ret < 0) {
        printf("[%s][%d] fprintf to file %s failed \n", __FUNCTION__, __LINE__, filename.c_str());
        return;
    }
    for (i = 0; i < ysize; i++) {
        ret = fwrite(buf + i * wrap, 1, xsize, f);
        if (ret < 0) {
            printf("[%s][%d] fwrite to file %s failed \n", __FUNCTION__, __LINE__, filename.c_str());
            return;
        }
    }
    ret = fclose(f);
    if (ret < 0) {
        printf("[%s][%d] fclose file %s failed \n", __FUNCTION__, __LINE__, filename.c_str());
        return;
    }
}

static int decode(AVCodecContext *dec_ctx, AVFrame *frame, AVPacket *pkt, const char *filename)
{
    int ret;

    ret = avcodec_send_packet(dec_ctx, pkt);
    if (ret < 0) {
        fprintf(stderr, "Error sending a packet for decoding\n");
        return ret;
    }

    while (ret >= 0) {
        ret = avcodec_receive_frame(dec_ctx, frame);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            return ret;
        } else if (ret < 0) {
            fprintf(stderr, "Error during decoding\n");
            return ret;
        }

        printf("saving frame %3d\n", dec_ctx->frame_number);
        fflush(stdout);

        /* the picture is allocated by the decoder. no need to free it */
        std::ostringstream sstream;
        sstream << filename << "-" << dec_ctx->frame_number << ".pgm";
        std::string saveFileName = sstream.str();

        pgmSave(frame->data[0], frame->linesize[0], frame->width, frame->height, saveFileName);
    }
}

int main(int argc, char **argv)
{
    const char *filename;
    const char *outfilename;
    const AVCodec *codec;
    AVCodecParserContext *parser;
    AVCodecContext *c= nullptr;
    FILE *f;
    AVFrame *frame;
    uint8_t inbuf[INBUF_SIZE + AV_INPUT_BUFFER_PADDING_SIZE];
    uint8_t *data;
    size_t dataSize;
    int ret;
    int eof;
    AVPacket *pkt;

    if (argc <= 2) {
        fprintf(stderr, "Usage: %s <input file> <output file>\n"
                "And check your input file is encoded by h264 please.\n", argv[0]);
        exit(0);
    }
    filename    = argv[1];
    outfilename = argv[2];

    pkt = av_packet_alloc();
    if (!pkt) {
        exit(1);
    }

    /* set end of buffer to 0 (this ensures that no overreading happens for damaged h264 streams) */
    memset(inbuf + INBUF_SIZE, 0, AV_INPUT_BUFFER_PADDING_SIZE);

    /* find the h264 video decoder */
    codec = avcodec_find_decoder(AV_CODEC_ID_H264);
    if (!codec) {
        fprintf(stderr, "Codec not found\n");
        exit(1);
    }

    parser = av_parser_init(codec->id);
    if (!parser) {
        fprintf(stderr, "parser not found\n");
        exit(1);
    }

    c = avcodec_alloc_context3(codec);
    if (!c) {
        fprintf(stderr, "Could not allocate video codec context\n");
        exit(1);
    }

    /* open it */
    if (avcodec_open2(c, codec, NULL) < 0) {
        fprintf(stderr, "Could not open codec\n");
        exit(1);
    }

    f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Could not open %s\n", filename);
        exit(1);
    }

    frame = av_frame_alloc();
    if (!frame) {
        fprintf(stderr, "Could not allocate video frame\n");
        exit(1);
    }

    do {
        /* read raw data from the input file */
        dataSize = fread(inbuf, 1, INBUF_SIZE, f);
        if (ferror(f)) {
            break;
        }
        eof = !dataSize;

        /* use the parser to split the data into frames */
        data = inbuf;
        while (dataSize > 0 || eof) {
            ret = av_parser_parse2(parser, c, &pkt->data, &pkt->size,
                                   data, dataSize, AV_NOPTS_VALUE, AV_NOPTS_VALUE, 0);
            if (ret < 0) {
                fprintf(stderr, "Error while parsing\n");
                exit(1);
            }
            data      += ret;
            dataSize -= ret;

            if (!pkt->size && eof) {
                break;
            }

            if (!pkt->size) {
                continue;
            }

            ret = decode(c, frame, pkt, outfilename);
            if (ret < 0) {
                fprintf(stderr, "Error while decoding\n");
                exit(1);
            }
        }
    } while (!eof);

    /* flush the decoder */
    decode(c, frame, NULL, outfilename);

    ret = fclose(f);
    if (ret < 0) {
        printf("[%s][%d] fclose file %s failed \n", __FUNCTION__, __LINE__, filename);
    }

    av_parser_close(parser);
    avcodec_free_context(&c);
    av_frame_free(&frame);
    av_packet_free(&pkt);

    return 0;
}
