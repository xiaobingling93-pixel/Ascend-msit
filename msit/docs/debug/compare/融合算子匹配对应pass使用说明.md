# 融合算子匹配对应Pass

## 1 简介

融合算子匹配对应Pass功能主要用于提供融合算子对应的Pass相关信息，快速定位有问题的精度Pass。
此功能使用包含两步：

1. 使用CANN包中 `msaccucmp.py` 工具的比对功能，得到算子融合前后的比对结果csv；
2. 基于比对结果csv，生成一个新的csv，将融合算子关联的融合Pass及融合前后未匹配的节点记录下来。

## 2 使用方式

### 前置准备
1. 请参见链接[安装msit](../../install/README.md)。

2. 使用msit debug dump功能

    分别进行以下两种情况的 Dump：

    - 正常推理：直接运行推理任务并生成 Dump 数据。
    - 关闭融合：使用 --fusion-switch-file 参数传递关闭融合规则的文件，运行推理任务并生成 Dump 数据。关闭融合规则文件的格式和内容可以参考[关闭融合规则文件说明](https://www.hiascend.com/document/detail/zh/canncommercial/80RC1/developmentguide/appdevg/aclcppdevg/aclcppdevg_000105.html)。

    具体使用参考[dump使用方法](../dump/README.md)

### dump文件比对

通过调用 CANN 包中的 `msaccucmp.py` 工具，对正常推理和关闭融合的 Dump文件进行比对。
参考使用命令如下：

```bash
python /home/HwHiAiUser/Ascend/ascend-toolkit/latest/toolkit/tools/operator_cmp/compare/msaccucmp.py compare \
 -m /home/HwHiAiUser/dump/{timestamp}/dump_data/npu/{timestamp}/ge_default_xxx/1/0 \  
 -g /home/HwHiAiUser/dump_without_fusion/{timestamp}/dump_data/npu/{timestamp}/ge_default_xxx/1/0 \ 
 -f /home/HwHiAiUser/dump/{timestamp}/mode/ge_proto_00000001_graph_11_Build.json \ 
 -cf /home/HwHiAiUser/dump_without_fusion/{timestamp}/mode/ge_proto_00000001_graph_11_Build.json \ 
 -out /home/HwHiAiUser/output
```
参数说明：

| 参数 | 说明 |
| --- | --- |
| `-m` | 指定正常推理 Dump 的最底层数据目录路径 |
| `-g` | 指定关闭融合后 Dump 的最底层数据目录路径 |
| `-f` | 指定正常推理 Dump 的 Model 文件夹下的 JSON 文件路径 |
| `-cf` | 指定关闭融合后 Dump 的 Model 文件夹下的 JSON 文件路径 |
| `-out` | 指定输出路径 |

输出结果说明参考[对比结果分析步骤](../../../examples/cli/debug/compare/result_analyse/README.md)

### 匹配融合算子对应的Pass
1. 准备脚本 match_fusion.py。

    以下是一个示例脚本，用于读取比对结果的 CSV 文件和融合 JSON 文件，并生成包含融合算子对应 Pass 信息的 CSV 文件。
    ```
    import argparse

    from components.debug.compare.fusion_pass_cmp.get_fusion_pass import fusion_pass_analysis
    from components.debug.compare.msquickcmp.common.args_check import str2bool


    def main():
        parser = argparse.ArgumentParser(description='Read compare output and fusion json file')
        parser.add_argument('-cp', '--csv-path', dest='csv_result_path', type=str,
                            help='比对结果的 CSV 文件路径', required=True)
        parser.add_argument('-jp', '--fusion-json-path', dest='fusion_json_path', type=str,
                            help='融合 JSON 文件路径', required=True)
        parser.add_argument('-o', '--out-csv-path', dest='out_csv_path', type=str,
                            help='输出 CSV 文件路径', required=True)
        parser.add_argument('-fn', '--fusion-node-switch', dest='fusion_node_switch', type=str2bool,
                            nargs='?', const=True, default=True, help='是否仅输出融合算子（默认为 True）')
        args = parser.parse_args()
        fusion_pass_analysis(args.csv_result_path, args.fusion_json_path, args.out_csv_path, args.fusion_node_switch)

    if __name__ == "__main__":
        main()
    ```
    参数说明：

   |参数名称|参数说明|
   |--|--|
   |`--csv-path`|指定比对结果的 CSV 文件路径|
   |`--fusion-json-path`|指定正常推理 Dump 的 Model 文件夹下的 JSON 文件路径|
   |`--out-csv-path`|指定一个输出的 CSV 文件路径|
   |`--fusion-node-switch`|是否仅输出融合算子（默认为 True）|

2. 调用上述脚本运行
    ``` bash
    python match_fusion.py -cp /home/HwHiAiUser/output/result_{timestamp}.csv \ 
    -jp /home/HwHiAiUser/dump/{timestamp}/mode/ge_proto_00000001_graph_11_Build.json \
    -o /home/HwHiAiUser/output/fusion_pass_out.csv
    ```

## 3 输出结果说明

- **比对结果**：在指定的输出文件 `xxx.csv` 中，输出正常推理情况下算子的output数据与关闭融合output数据的比对情况，并给出融合算子对应的PassName。
* 比对结果示例及说明如下：

| 关注项 / 结果项 | OpType | NPUDump                             | DataType | Address | GroundTruth  | TensorIndex     | Shape          | CosineSimilarity                              | ... | MeanRelativeError                             | PassName                                         | MatchError  |
|-----------|--------|-------------------------------------|----------|---------|--------------|-----------------|----------------|-----------------------------------------------|-----|-----------------------------------------------|--------------------------------------------------|-------------|
| 示例值       | Conv2D | Conv2D                              | float32  | NaN     | Conv2D,Pad   | Conv2D:output:0 | [1,64,112,112] | 1                                             | ... | 0.000364                                      | "{'PadConv2dFusionPass', 'AABiasaddConvFusion'}" |             |
| 说明        | 算子类型   | 正常推理模型中的算子，由于融合规则，可能会对应多个关闭融合情况下的算子 | 数据类型     | -       | 关闭融合情况下的模型算子 | -               | -              | 各类误差比对类型结果，主要需要看是否某一项超过精度阈值(即某项异常)，若超过则需要重点关注 | -   | 各类误差比对类型结果，主要需要看是否某一项超过精度阈值(即某项异常)，若超过则需要重点关注 | 融合算子对应的融合规则                                      | 标识未匹配上的融合算子 |

 具体阈值可参考 [精度比对结果：比对结果分析](../../../examples/cli/debug/compare/result_analyse/README.md#比对结果分析)
