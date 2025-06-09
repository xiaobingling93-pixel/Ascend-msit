# msit debug opcheck功能使用指南
## 简介
opcheck（精度预检）提供了传统小模型场景下kernel级别的精度预检功能，支持对经过GE推理后dump落盘数据进行算子精度预检，检测kernel级别的算子精度是否达标。

**注:** 目前只支持TensorFlow2.6.5，未来会支持更多框架

## 环境准备
### 1. TensorFlow2.6.5
```
pip3 install tensorflow==2.6.5 --no-index --find-links  https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/MindX/OpenSource/python/index.html --trusted-host ascend-repo.obs.cn-east-2.myhuaweicloud.com
```
### 2.PyTorch
```
pip install torch==1.11.0
```
### 3.工具安装
- 工具安装请见 [msit一体化工具使用指南](../../install/README.md)

## 使用方法
### 输入数据
使用`msit debug dump -m /home/HwHiAiUser/prouce_data/resnet_offical_saved_model -dp npu -is "模型的输入shape" -c /usr/local/Ascend/ascend-toolkit/latest -o /home/HwHiAiUser/result/test` dump模型推理过程的tensor数据
- `msit debug dump`相关参数使用参考[saved_model dump功能使用参考](../../../examples/cli/debug/dump/06_saved_model/README.md)
- tensor数据会生成在-o参数指定的路径下

### 使用入口
预检功能可以直接通过msit命令行形式启动精度对比。启动方式如下：
命令示例，**其中路径需使用绝对路径**
```sh
msit debug opcheck -i /home/HwHiAiUser/result/test/{timestamp} -o /home/HwHiAiUser/result/test/opcheck_test
```

### 参数说明
| 参数名                      | 描述                                                         | 是否必选 |
| --------------------------- | ------------------------------------------------------------ | -------- |
| --input, -i                 | 由msit debug dump 命令落盘的tensor数据文件夹，示例：{output_path}/{timestamp} | 是       |
| --output, -o                | 精度预检结果的保存路径，默认为当前路径               | 否       |
| --help, -h              | 命令行参数帮助信息| 否       |

### 输出结果说明
输出结果生成在result_{timestamp}.xlsx文件中，包含两张表格分别为opcheck_results及addition_failed_cases，分别包含预检成功及失败的算子。
#### 结果表格各列说明
| 表头                    | 说明                                     |
|-----------------------|----------------------------------------|
| op_type               | 算子类型，格式为以list包含的算子类名                   |
| op_name               | 算子名称，以'_'分隔的算子拓扑结构名                    |
| op_param              | 算子参数                                   |
| tensor_path           | 算子输入intensor及输出outtentor的文件名称          |
| out_tensor_id         | 算子输出outtensor的序号（部分算子输出可能有多个outtensor） |
| rel_precision_rate(%) | 实际的精度通过率（使用相对误差，全部通过则为100%）            |
| max_rel_error         | 最大的相对误差值                               |
| abs_precision_rate(%) | 实际的绝对误差精度通过率                           |
| max_abs_error         | 最大的绝对误差值                               |
| cosine_similarity     | 余弦相似度                                  |
| kl_divergence         | kl散度                                   |
| fail_reason           | 失败原因，包括精度未达到给定的标准或算法执行失败原因             |

相关计算公式为：
```
相对误差：rel_error = abs(actual_output - golden_output) / abs(golden_output)
精度通过率：rel_precision_rate = sum(rel_error <= etol) / size(rel_error) * 100
精度比对结果：precision_result = bool(rel_precision_rate >= pass_rate)
```
### 目前精度预检算子支持情况
| 算子名称(A-G)     | 算子名称(H-M)  | 算子名称(N-R)  | 算子名称(S-Z)   |
|---------------|------------|------------|-------------|
| Add           | LogicalOr  | Pad        | SoftMaxV2   |
| Adds          | LogicalAnd | PadD       | Sub         |
| BatchNorm     | Less       | Pack       | Sigmoid     |
| BatchMatMulV2 | Mul        | ReduceMean | StrideSlice |
| BiasAdd       | Minimum    | Rsqrt      | Select      |
| BNInfer       | Mul        | Relu       | Tile        |
| Cast          | MatMulV2   | ReduceSum  | Tanh        |
| ConcatV2      |            |            | Transpose   |
| Conv2D        |            |            |             |
| ClipByValue   |            |            |             |
| GatherV2      |            |            |             |

## FAQ
1. 为什么需要使用PyTorch 1.11.0版本？

标杆实现采用PyTorch 1.11.0版本，如果使用其他版本的PyTorch，可能导致算子实现不同

2. 为什么BatchNorm算子预检会出现异常？

BatchNorm算子的标杆实现采用了PyTorch的torch.ops.aten.native_batch_norm实现，该实现在计算过程中可能会出现inf/nan等异常场景