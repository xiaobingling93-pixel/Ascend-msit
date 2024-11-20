
# 基于MindIE-Torch 精度比对场景

## 1. 相关依赖

- CANN（8.0RC3及以上）
- MindIE（1.0RC3及以上）

## 2. msit安装

- msit安装请参考[一体化安装指导](/msit/docs/install/README.md)

## 3. Dump 数据 

### NPU数据Dump

- 准备 `bert_inference.py`，为模型的前向推理脚本

- 执行 `msit debug dump --exec "python bert_inference.py" [--option]` Dump NPU数据 

  ```sh
  msit debug dump --output [/path/to/dump] --exec "python bert_inference.py" --operation-name MatMulv2_1,trans_Cast_0
  ```

- 参数说明 

| 参数名          | 描述                                                                                                                             | 必选 |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------- | ---- |
| --exec | MindIE-Torch推理脚本的执行命令                                                                                   | 是   |
| --output | 指定Dump数据输出路径，默认为当前路径                                 | 否   | 
| -opname, --operation-name | 需要Dump的算子，默认为 all，表示会对模型中所有 op 进行 Dump，其中元素为MindIE-Torch算子类型，若设置 operation-name，只会 Dump 指定的 op                                | 否   | 

### CPU/GPU数据Dump

- CPU/GPU数据Dump请参考[CPU/GPU数据Dump](/msit/docs/llm/工具-DUMP在线推理数据使用说明.md)

#### 注意

- NPU数据Dump时请在线进行推理，若save模型后load再进行推理会使得json文件缺少必要信息，无法进行compare

- NPU数据Dump后会在当前路径下生成两个算子类型映射关系文件，名称为：`mindie_torch_op_mapping.json` 和 `mindie_rt_op_mapping.json` 

## 4. Compare 精度对比 

 - 执行 `msit debug compare --golden-path [/path/to/cpu/dumpdata] --my-path [/path/to/npu/dumpdata] --output [path/to/csv] --ops-json [path/to/json]`，输出比对结果 csv 文件

### 参数说明

| 参数名                 | 描述                                                         | 必选 |
| ---------------------- | ------------------------------------------------------------ | -------- |
| --golden-path, -gp     | CPU/GPU Dump数据根路径                   | 是       |
| --my-path, -mp         | NPU Dump数据根路径                         | 是       |
| --ops-json | 运行 `msit debug dump` 时产生的算子映射关系文件路径，通常在当前文件夹下 | 是       |
| --output, -o           | 比对结果csv的输出路径，默认为当前路径              | 否       |

## 注意

- MindIE场景算子精度对比目前仅支持TorchScript路线