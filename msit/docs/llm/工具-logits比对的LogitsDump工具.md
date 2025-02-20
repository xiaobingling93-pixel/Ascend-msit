# msIT Logits Dump 工具
# 背景

在大模型的精度测评工作中，不同设备之间产生的推理结果的精度会略有不同。对精度测评结果使用Bad Case分析工具后，可以获得同一模型同一数据集在不同环境下精度测评之后的`bad case`集合。针对这些错例集合，可以通过dump部分token的logits数据进一步比对分析，证明错例的真伪。因此当前Logits Dump工具提供`NPU`和`GPU`环境下对模型推理的logits结果落盘的功能。


# 介绍

Logits Dump工具主要功能是对模型推理过程中的Logits进行落盘。首先分析Bad Case结果，在内存中将该结果与全量数据集取交集，在推理过程中仅推理这些错例。在错例集推理过程中，保存每个错例case的固定长度token的Logits结果。

**说明** ：模型推理过程依赖modeltest工具，请在使用Logits Dump工具前确保modeltest可用。

## 参数说明

由于工具底层调用了modeltest工具，所以需要传入modeltest必须的参数，且参数配置遵循modeltest规则。

| 参数                  | 别名   | 描述                                                                | 是否必选 |
| --------------------- | ----- | ------------------------------------------------------------------- | ------- |
| --model-config-path   | -mcp  | 模型推理的模型配置文件(modeltest推理必须文件，详情请见modeltest资料)    | 是      |
| --task-config-path    | -tcp  | 模型推理的数据集配置文件(modeltest推理必须文件，详情请见modeltest资料)  | 是     |
| --bad-case-result     | -bcr  | Bad Case分析工具的结果文件，存储着两种场景下的精度测评结果的错例集       | 是     |
| --device              | -d    | 模型推理选择的设备，默认0卡单卡                                       | 否     |
| --output-dir          | -o    | Logits Dump结果的输出文件夹，默认当前文件夹下的output文件夹下          | 否     |
| --token-range         | -tr   | 需要dump的前多少条token的logits                                      | 否     |
| --help                | -h    | 命令行帮助信息                                                      | 否      |

**端口**：由于modeltest在`NPU `上多卡推理时，需要使用`torchrun`命令，并需要设定端口号。当前工具硬编码使用`9826`端口号，请在使用本工具时，确认该端口未被占用。

## 使用方法

```shell
msit llm logitsdump -mcp {model_config_path} -tcp {task_config_path} -bcr {bad_case_csv_path} [ -d {device_num} -o {output_path} -tr {int} ]
```

传入的两个`config_path`用于modeltest的推理，`bad_case_path`是Bad Case分析工具的输出结果，在当前工具中用于在全集数据集中筛选错例。可选参数可以配置推理的设备、输出文件夹以及dump数据范围。

**说明**：
  - 模型推理过程中请按照modeltest工具的数据集准备方法，将数据集配置到当前指令运行的文件夹下，或者在task_config文件中设定本地数据集地址参数。
  - humanevalx数据集是代码评测类数据集，推理时额外需要modeltest的tools文件夹，需要手动将MindIE-LLM仓库下的examples/atb_models/tests/modeltest/tools/文件夹，复制到当前运行指令的文件夹下，或者在examples/atb_models/tests/modeltest/路径下运行当前工具。
  - 当设定的`--token-range`超过output token的个数时，仅会dump所有token个数的Logits。

# 示例
下列示例展示了如何通过命令行进行Logits Dump工作。

利用Bad Case分析工具获得BoolQ数据集在不同环境下的Bad Case结果文件，例如：`msit_bad_case_result_20250117134959.csv`

```shell
msit llm logitsdump -mcp ./config/model/llama3-8b.yaml -tcp ./config/task/boolq.yaml -bcr ./msit_bad_case_result_20250117134959.csv -d 0,1,2,3 -o ./output -tr 10
```
上述命令行配置了两个modeltest推理所需的yaml配置文件地址，以及Bad Case结果地址，同时设定了在`0,1,2,3`四卡上进行推理和落盘前10个token的Logits数据。

获得存放在类似`./output/data/{device}/precision_test/boolq/{}/llama3-8b/logits/`文件夹路径下的所有Logits结果。其中存放的文件名类似于：`logits_{task_name}_{batch_id}_{token_id}.pth`，每个文件中保存了一个batch的单个token的Logits结果。
