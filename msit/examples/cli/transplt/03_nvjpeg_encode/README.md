## Nvjpeg 图片编码迁移
- 将使用 nvjpeg 的图片编码样例迁移到昇腾平台使用 ACL 接口实现，将一帧 yuv 格式的视频流输入编码为 jpg 格式并保存到文件。

## 准备测试图片
- 昇腾平台视频流使用的 YUV420 格式编码方式为 semi planar，即 YUV420SP，nvjpeg 使用的是 planar，即 YUV420，准备同一张图片的两种编码格式数据
- 使用 `scikit-image` 中的测试图片，也可使用自行获取的图片
  ```py
  import cv2
  import numpy as np
  from skimage.data import chelsea

  height, width = 480, 640
  image = cv2.resize(chelsea()[:300, :400], (width, height))
  yuv_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV_I420)
  yuv_data = yuv_image.flatten().astype('uint8')
  yuv_data.tofile('sample_{}_{}_planar.yuv'.format(height, width))

  blocks = height * width
  cc = np.stack([yuv_data[blocks:blocks + blocks // 4], yuv_data[blocks + blocks // 4:]]).T.flatten()
  semi_yuv_data = np.concatenate([yuv_data[:blocks], cc.flatten()])
  semi_yuv_data.tofile('sample_{}_{}_semiplanar.yuv'.format(height, width))
  ```
## NVJpeg encode 编译执行
- 在 GPU 环境上，指定 `CUDA_PATH` 为实际 cuda 库位置
  ```sh
  CUDA_PATH=/usr/local/cuda-11.8
  g++ -g -m64 nvjpeg_encode.cpp -o nvjpeg_encode -lnvjpeg -I $CUDA_PATH/targets/x86_64-linux/include \
  -ldl -lrt -pthread -lcudart -L$CUDA_PATH/lib64
  ```
- **执行** 指定输入图片以及图片高、宽，输出文件为 `sample_nvjpeg.jpg`
  ```sh
  ./nvjpeg_encode sample_480_640_planar.yuv 480 640
  # Writing JPEG file: sample_nvjpeg.jpg, length: 164466
  ```
## msIT transplt 迁移分析
- 安装 msit 工具后，针对待迁移项目执行 transplt 迁移分析，当前输出结果中对应的 `AscendAPI` 为 ACL V2 接口
  ```sh
  msit transplt -s .
  # INFO - scan_api.py[123] - Scan source files...
  # ...
  # INFO - csv_report.py[46] - Report generated at: ./output.xlsx
  # INFO - scan_api.py[113] - **** Project analysis finished <<<
  ```
- 最终分析结果文件位于当前执行路径下 `./output.xlsx`，该结果中重点关注有对应关系的接口，并参照 `AscendAPILink` 中相关接口说明辅助完成迁移
- 输出结果的表格文件具体字段说明以及使用方法参照 [01_basic_usage](../01_basic_usage) 及 [02_resnet50_inference](../02_resnet50_inference)
## ACL Encode
- ACL 接口目前分为 V1 和 V2，昇腾 310 AI 处理器上，当前仅支持 V1 版本的媒体数据处理接口，昇腾310P AI处理器上，支持V1和V2两个版本的媒体数据处理接口，接口间 v1 到 V2 迁移参照 [昇腾310 AI处理器媒体数据处理V1->昇腾310P AI处理器媒体数据处理V2迁移指引](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/70RC1alpha001/infacldevg/aclcppdevg/aclcppdevg_000165.html)
- **ACL V1 接口编译执行** 迁移完成后在 Ascend 310/310P 上执行，`ASCEND_TOOLKIT_HOME` 为实际 CANN 目录
  ```sh
  g++ acl_encode_v1.cpp -o acl_encode_v1 -I $ASCEND_TOOLKIT_HOME/runtime/include -DENABLE_DVPP_INTERFACE \
  -L $ASCEND_TOOLKIT_HOME/runtime/lib64/stub/ -lpthread -lstdc++  -lascendcl -lacl_dvpp

  ./acl_encode_v1 sample_480_640_semiplanar.yuv 480 640
  # Open device 0 success
  # Writing JPEG file: sample_acl_v1.jpg, length: 164800
  ```
- **ACL V2 接口编译执行** 迁移完成后在 Ascend 310P 上执行，`ASCEND_TOOLKIT_HOME` 为实际 CANN 目录，
  ```sh
  g++ acl_encode_v2.cpp -o acl_encode_v2 -I $ASCEND_TOOLKIT_HOME/runtime/include -DENABLE_DVPP_INTERFACE \
  -L $ASCEND_TOOLKIT_HOME/runtime/lib64/stub/ -lpthread -lstdc++  -lascendcl -lacl_dvpp -lacl_dvpp_mpi

  ./acl_encode_v2 sample_480_640_semiplanar.yuv 480 640
  # Open device 0 success
  # Writing JPEG file: sample_acl_v2.jpg, length: 164800
  ```
  Ascend310 上将报错 `acldvppMalloc malloc device data buffer failed, aclRet is -1610448875`
