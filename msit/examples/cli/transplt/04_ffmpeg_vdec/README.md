# FFmpeg 视频解码迁移分析案例
- 本示例将FFmpeg视频解码应用工程迁移到昇腾平台

## 准备测试视频

下载样例视频[dvpp_vdec_h264_1frame_bp_51_1920x1080.h264](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/data/dvpp_sample_input_data/dvpp_vdec_h264_1frame_bp_51_1920x1080.h264)，该视频只包含一帧1920*1080尺寸的图片。

## 运行原始FFmpeg工程

> 建议在FFmpeg 4.2以上版本运行该工程

### 安装FFmpeg

这里以Ubuntu为例，如果是Ubunt20.04以上版本，可以直接用如下命令安装：

```shell
sudo apt install ffmpeg libavformat-dev libavcodec-dev libavfilter-dev libavdevice-dev libavresample-dev libavutil-dev -y
```

如果是Ubuntu18.04及以下版本，则需要下载源码来编译安装，从[这里](https://launchpadlibrarian.net/590398075/ffmpeg_4.4.1.orig.tar.xz)x下载FFmpeg4.4.1版本的源码包。由于本用例是解码.h264视频，因此还需要安装h264包，可以运行如下命令来安装

```shell
sudo apt install libx264-dev -y
```

然后在FFmpeg的源码目录，运行如下命令来编译安装FFmpeg：

```shell
> ./configure --enable-shared --disable-static --disable-doc --enable-gpl --enable-libx264
> make
> sudo make install
```

### 测试FFmpeg是否安装成功

在样例视频的目录下运行如下命令：

```shell
ffmpeg -i dvpp_vdec_h264_1frame_bp_51_1920x1080.h264 test_video.mp4
```

如果命令运行成功且将h264视频转换成了mp4视频，则证明FFmpeg安装成功。

### 编译并运行FFmpeg应用工程

在`<ait工程目录>/ait/examples/cli/transplt/04_ffmpeg_vdec/ffmpeg`下使用如下命令编译cpp源码文件：

```shell
g++ -std=c++11 ffmpeg_decode.cpp -lavcodec -lavdevice -lavfilter -lavformat -lavutil -lswresample -lswscale -o ffmpeg_decode
```

然后运行编译后的可执行文件：

```shell
./ffmpeg_decode dvpp_vdec_h264_1frame_bp_51_1920x1080.h264 dvpp_vdec_h264_1frame_bp_51_1920x1080_decoded
```

如果运行成功，会在当前目录下生成一个文件名为`dvpp_vdec_h264_1frame_bp_51_1920x1080_decoded-1.pgm`的文件，该文件为解码后的灰度pgm格式图片。

## AIT Transplt 迁移分析

  - 安装 ait 工具后，针对待迁移项目执行 transplt 迁移分析
  ```sh
  ait transplt -s <ait工程目录>/ait/examples/cli/transplt/04_ffmpeg_vdec/ffmpeg
  ```
  输出示例如下
  ```sh
  INFO - scan_api.py[123] - Scan source files...
  ...
  INFO - csv_report.py[46] - Report generated at: ./output.xlsx
  INFO - scan_api.py[113] - **** Project analysis finished <<<
  ```

  最终分析结果文件位于`<ait工程目录>/ait/examples/cli/transplt/04_ffmpeg_vdec/ffmpeg`路径下 `./output.xlsx`，该结果中重点关注有对应关系的接口，并参照 `AscendAPILink` 中相关接口说明辅助完成迁移。

## 迁移到昇腾ACL DVPP 图像处理
- **完成该部分迁移，可使用`ACL DVPP` 图像处理在昇腾 NPU 上执行视频解码**

在FFmpeg中，完整的视频解码流程主要包含了以下几个部分：1）函数库初始化；2）创建并初始化解码器；3）读取视频帧数据；4）发送视频帧到解码器并从解码器接收解码后的数据。

- **函数库初始化**
  
`FFmpeg 4.2`及之前的版本中在初始化时需要调用函数`av_register_all`，只有调用了该函数，才能正常使用`FFmpeg`的各项功能。但是在新版本的`FFmpeg`中，该函数已经被废弃，无需再调用。
而`ACL DVPP`在使用时仍需要调用下列初始化函数

| AscendAPI       | Description                          |
| --------------- | ------------------------------------ |
| aclInit         | AscendCL初始化函数                   |
| aclrtSetDevice  | 指定当前进程或线程中用于运算的Device |
| hi_mpi_sys_init | 媒体数据处理系统底层初始化           |
  
- **创建并初始化解码器**

FFmpeg初始化解码器的相关API如下

| AccAPI                 | Description                     |
| ---------------------- | ------------------------------- |
| av_packet_alloc        | 分配packet结构体                |
| avcodec_find_decoder   | 查找解码器                      |
| av_parser_init         | 初始化解码器                    |
| avcodec_alloc_context3 | 分配一个解码context             |
| avcodec_open2          | 用指定的解码器初始化解码context |

对应的ACL DVPP初始化解码器相关API如下

| AscendAPI                     | Description                                              |
| ----------------------------- | -------------------------------------------------------- |
| aclrtCreateContext            | 在当前进程或线程中显式创建一个Context                    |
| hi_mpi_vdec_create_chn        | 根据设置的通道属性创建解码通道                           |
| hi_mpi_vdec_get_chn_param     | 修改通道参数前可先调用此接口获取通道参数                 |
| hi_mpi_vdec_set_chn_param     | 改变通道参数的时候调用此接口，设置解码通道参数           |
| hi_mpi_vdec_start_recv_stream | 创建通道之后，启动解码之前，解码器开始接收用户发送的码流 |

- **读取视频帧数据**

FFmpeg提供了一个API `av_parser_parse2`用于从数据流中切分视频帧，其读取视频帧数据相关API如下

| AccAPI           | Description                        |
| ---------------- | ---------------------------------- |
| av_frame_alloc   | 分配内存                           |
| fread            | 从文件中读取二进制数据（系统函数） |
| av_parser_parse2 | 切分视频帧数据                     |

而ACL DVPP没有提供类似的API，需要用户根据不同的视频格式手动切分视频帧，其读取视频帧数据相关API如下

| AccAPI             | Description                        |
| ------------------ | ---------------------------------- |
| hi_mpi_dvpp_malloc | 分配Host内存                       |
| aclrtMemcpy        | 将内存从Host侧拷贝到Device侧       |
| fread              | 从文件中读取二进制数据（系统函数） |
| get_every_frame    | 切分视频帧函数（用户自定义函数）   |

- **发送视频帧到解码器并从解码器接收解码后的数据**
  
FFmpeg的解码相关API如下：

| AccAPI                | Description                           |
| --------------------- | ------------------------------------- |
| avcodec_send_packet   | 发送视频帧数据到解码器                |
| avcodec_receive_frame | 从解码器接收解码后的帧数据            |
| 保存文件API           | 保存解码后图片到pgm文件（用户自定义） |

而DVPP的视频解码部分使用的是多线程运行方式，其中一个线程负责向Device发送待解码的视频帧，另一个线程负责从Device接收解码后的数据。因此需要使用pthread_create函数创建2个不同的子线程来完成视频解码工作。

其中发送待解码视频帧的线程相关调用API如下：

| AscendAPI                         | Description                                          |
| --------------------------------- | ---------------------------------------------------- |
| aclrtSetCurrentContext            | 设置线程的Context                                    |
| 上面步骤中的读取视频帧数据相关API | 读取视频帧数据                                       |
| hi_mpi_vdec_send_stream           | 解码前，向解码通道发送码流数据及存放解码结果的buffer |
| hi_mpi_dvpp_free                  | 释放Device上的内存                                   |

而接收已解码视频帧的线程相关调用API如下：

| AscendAPI                 | Description                                |
| ------------------------- | ------------------------------------------ |
| aclrtSetCurrentContext    | 设置线程的Context                          |
| hi_mpi_vdec_get_frame     | 解码后，获取解码通道的解码图像及输入Stream |
| 保存文件API               | 保存解码后图片到pgm文件（用户自定义）      |
| hi_mpi_vdec_release_frame | 解码之后，释放资源                         |

- **资源释放**
  
FFmpeg在解码完成后，资源释放相关API如下：

| AccAPI               | Description       |
| -------------------- | ----------------- |
| av_parser_close      | 关闭解码器        |
| avcodec_free_context | 释放Context的资源 |
| av_frame_free        | 释放视频帧结构体  |
| av_packet_free       | 释放packet结构体  |

DVPP在解码完成后，资源释放相关API如下：

| AscendAPI           | Description                                                  |
| ------------------- | ------------------------------------------------------------ |
| hi_mpi_sys_exit     | 用户处理完图像、视频等媒体数据之后，需先调用本接口对媒体数据处理系统底层进行去初始化 |
| aclrtDestroyContext | 销毁一个Context，释放Context的资源                           |
| aclrtResetDevice    | 复位当前运算的Device，释放Device上的资源                     |
| aclFinalize         | AscendCL去初始化函数，用于释放进程内的AscendCL相关资源       |
  
- **编译执行**

迁移后的代码实现可以参考[acl_decode.cpp](https://gitee.com/ascend/ait/tree/master/ait/examples/cli/transplt/04_ffmpeg_vdec/acl/acl_decode.cpp)。另外还有一个简化版本的实现[acl_decode_simplified.cpp](https://gitee.com/ascend/ait/tree/master/ait/examples/cli/transplt/04_ffmpeg_vdec/acl/acl_decode_simplified.cpp)，简化版本阉割了多线程解码能力，并且删除了许多错误检查代码，简化版代码只是为了让用户更方便的理解ACL视频解码的流程，实际运行过程中可能会存在未知问题。
迁移完成后在`<ait工程目录>/ait/examples/cli/transplt/04_ffmpeg_vdec/acl`下使用 `g++` 编译源码文件，也可自行编写 `cmake` 文件，修改调整至可编译通过并正确执行。

> **注意**： 迁移后的代码使用了ACL数据媒体处理V2版本的接口，该接口目前只支持昇腾310P AI处理器，请在310P的机器上编译运行迁移后的代码。
  
```sh
g++ -O3 -std=c++11 acl_decode.cpp -o acl_decode -I$ASCEND_TOOLKIT_HOME/runtime/include/acl -I$ASCEND_TOOLKIT_HOME/runtime/include/acl/dvpp -L $ASCEND_TOOLKIT_HOME/runtime/lib64/stub -lacl_dvpp_mpi -lascendcl -lpthread -lstdc++

./acl_decode dvpp_vdec_h264_1frame_bp_51_1920x1080.h264 dvpp_vdec_h264_1frame_bp_51_1920x1080_decoded
```
输出示例如下：
```sh
[hi_dvpp_init][676] aclInit Success.
[hi_dvpp_init][684] aclrtSetDevice 0 Success.
[hi_dvpp_init][693] aclrtCreateContext Success
[hi_dvpp_init][704] Dvpp system init success
[send_stream][532] Chn 0 send_stream Thread Exit
[get_pic][574] Chn 0 GetFrame Success, Decode Success[1]
[get_pic][608] Chn 0 get_pic Thread Exit
[hi_dvpp_deinit][730] Dvpp system exit success  
```

> 其中`$ASCEND_TOOLKIT_HOME`是CANN包路径，假设CANN包安装在`/usr/local/Ascend`，该环境变量可以通过`source /usr/local/Ascend/ascend-toolkit/set_env.sh`来自动设置。