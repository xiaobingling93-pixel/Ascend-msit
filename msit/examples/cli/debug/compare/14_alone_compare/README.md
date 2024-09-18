# Alone Compare


## 介绍
指定dump数据进行compare精度对比功能可以通过msit命令行方式启动。


## 运行示例
- 命令示例 
- **注意：** 传入的-mp路径以及-gp路径必须为数据的上一层文件夹
  ```sh
  msit debug compare -mp /home/HwHiAiUser/onnx_prouce_data/resnet_offical_onnx -gp /home/HwHiAiUser/onnx_prouce_data/model/resnet_offical_om 
  --ops-json /home/HwHiAiUser/onnx_prouce_data/resnet_offical/ops.json -o /home/HwHiAiUser/result/test
  ```
  - `-mp, --my-path` 指定npu侧dump数据路径 
  - `-gp, --golden-path` 指定cpu侧dump数据路径
  - `--ops-json ` 指定cpu侧与npu侧算子的匹配规则json文件路径
  - `-o, –-output` (可选) 输出文件路径，默认为当前路径


### 输出结果说明和分析步骤参考

请参考：[对比结果分析步骤](../result_analyse/README.md)

