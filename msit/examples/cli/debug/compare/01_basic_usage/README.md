# Basic Usage


## 介绍
compare精度对比功能可以通过ait命令行方式启动。


## 运行示例
- **注意**：使用atc转换时确保转换后的om模型与源onnx模型输入数据类型一致。（如fp32输入的onnx模型atc转换时不能使用input_fp16_nodes参数）
- **不指定模型输入** 命令示例，**其中路径需使用绝对路径**
  ```sh
  ait debug compare -gm /home/HwHiAiUser/onnx_prouce_data/resnet_offical.onnx -om /home/HwHiAiUser/onnx_prouce_data/model/resnet50.om \
  -c /usr/local/Ascend/ascend-toolkit/latest -o /home/HwHiAiUser/result/test
  ```
  - `-om, –-om-model` 指定昇腾AI处理器的离线模型（.om）路径
  - `-gm, --golden-model` 指定模型文件（.pb或.onnx）路径
  - `-c，–-cann-path` (可选) 指定 `CANN` 包安装完后路径，不指定路径默认会从系统环境变量`ASCEND_TOOLKIT_HOME`中获取`CANN` 包路径，如果不存在则默认为 `/usr/local/Ascend/ascend-toolkit/latest`
  - `-o, –-output` (可选) 输出文件路径，默认为当前路径


### 输出结果说明和分析步骤参考

请参考：[对比结果分析步骤](../result_analyse/README.md)

