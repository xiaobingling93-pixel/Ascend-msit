# Specify Input Data


## 介绍

默认情况下，构造全为0的数据送入模型进行精度对比。可指定文件输入或者文件夹输入。
- **注意**：如果为文件夹输入，请确保使用的所有输入文件均为.bin后缀文件。

## 运行示例

**指定模型输入** 命令示例，**其中路径需使用绝对路径**
  ```sh
  ait debug compare -gm /home/HwHiAiUser/onnx_prouce_data/resnet_offical.onnx -om /home/HwHiAiUser/onnx_prouce_data/model/resnet50.om \
  -i /home/HwHiAiUser/result/test/input_0.bin -c /usr/local/Ascend/ascend-toolkit/latest -o /home/HwHiAiUser/result/test
  ```
  - `-i，–-input` 模型的输入数据路径，默认根据模型的 input 随机生成，多个输入以逗号分隔，例如：`/home/input_0.bin,/home/input_1.bin`，本场景会根据文件输入 size 和模型实际输入 size 自动进行组 Batch，但需保证数据除 batch size 以外的 shape 与模型输入一致
  ```sh
  ait debug compare -gm /home/HwHiAiUser/onnx_prouce_data/resnet_offical.onnx -om /home/HwHiAiUser/onnx_prouce_data/model/resnet50.om \
  -i /home/HwHiAiUser/result/test/input_0.npy -c /usr/local/Ascend/ascend-toolkit/latest -o /home/HwHiAiUser/result/test
  ```
- `-i，–-input` 如果存在npy文件，支持自动将npy文件转化为bin文件，而不需要手动转化为bin文件，例如：`/home/input_0.npy,/home/input_1.npy`
