# Caffe Dump

## 介绍
Caffe dump

## 使用示例一
- 环境中已安装 CANN 包以及 msit 工具
- 环境中安装 caffe，其中 `Ubuntu 18.04` 可通过 `apt install caffe-cpu` 安装
- 对于 Caffe 模型，目前不支持动态 shape 的模型比对。对于 `yolov2` / `yolov3` / `ssd` 等需要自定义实现层的模型，需要自行编译安装特定版本的 caffe
- 准备 Caffe 模型结构文件与权重文件，示例模型结构文件参照 [Caffe 模型结构文件示例](#caffe-模型结构文件示例)，以下使用该文件定义模型，并保存随机初始化的权重，实际使用中应使用已有的模型结构文件 `.prototxt` 与权重文件 `.caffemodel`
  ```py
  import caffe

  model_path = "caffe_demo.prototxt"
  weight_path = "caffe_demo.caffemodel"

  net = caffe.Net(model_path, caffe.TEST)
  net.save(weight_path)
  ``` 

- **精度比对调用**
  ```sh
  mkdir -p test  # dump 数据以及比对结果输出路径
  ASCEND_TOOLKIT=$HOME/Ascend/ascend-toolkit/latest  # 必须为可写的 CANN 包路径
  msit debug dump -m caffe_demo.prototxt -w caffe_demo.caffemodel -dp cpu -c $ASCEND_TOOLKIT -o ./test
  ```
- **输出目录结构** 参考 [01_basic_usage](../01_basic_usage/README.md)，其中 caffe 模型 dump 数据位于 `{output_path}/{timestamp}/dump_data/caffe/`
