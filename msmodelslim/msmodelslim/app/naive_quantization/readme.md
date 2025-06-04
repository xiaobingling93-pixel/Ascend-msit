# 一键量化使用说明
## 功能说明
一键量化功能为小白用户提供“开箱即用”的热门开源模型量化功能。提供全局地调用量化命令能力，通过指定必要的参数即可对目标原始权重做指定量化。

用户输入命令行，会根据指定要求在最佳实践库中匹配到最佳配置实施量化；如果最佳实践库中无法搜寻到最佳量化配置，则会申请是否采用默认配置实施量化。

## 工具使用说明
msmodelslim一键量化功能通过命令行方式启动，如果在工具安装时没有遇到 Python 的 PATH 变量问题，或已经手动将 Python 可执行文件的目录添加到 PATH 变量中，则可以使用如下命令格式运行：
``` bash
msmodelslim quant [ARGS]
```
例如，使用一键量化功能量化 Qwen2.5-7B-Instruct 模型，量化方式采用 w8a8 ，则量化命令如下：
``` bash
msmodelslim quant --model_path {模型路径} --save_path {保存路径} --device npu --model_type Qwen2.5-7B-Instruct --quant_type w8a8 --trust_remote_code True
```


## 接口说明
``` bash
#全局调用命令行
msmodelslim quant --model_path {模型路径} --save_path {量化权重保存路径} --device {量化设备} --model_type {模型名称} --config_path {指定配置路径} --quant_type {量化类型} --trust_remote_code {是否信任自定义代码}
```
|参数名称|解释|是否可选| 范围                                                                                    |
|--------|--------|--------|---------------------------------------------------------------------------------------|
|model_path|模型路径|必选| 类型：Str                                                                                |
|save_path|量化权重保存路径|必选| 类型：Str                                                                                |
|device|量化设备|可选| 1. 类型：Str <br>2. 可选值："cpu","npu" <br>3. 默认值为"npu"                                     |
|model_type|模型名称|必选| 1. 类型：Str <br>2. 大小写敏感，请参考下述支持的模型                                                     |
|config_path|指定配置路径|与"quant_type"二选一| 1. 类型：Str <br>2. 配置文件格式为yaml <br>3. 当前只支持最佳实践库中已验证的配置，若自定义配置，msmodelslim不为量化结果负责 <br> |
|quant_type|量化类型|与"config_path"二选一| w4a8, w4a16, w8a8, w8a8s, w8a8c8, w8a16                                               |
|trust_remote_code|是否信任自定义代码|可选| 1. 类型：Bool，默认值：False <br>2. 请确保加载的自定义代码文件的安全性，设置为True有安全风险。                           |

## 支持的模型
注意：
1. 命令行参数"--model_type"请严格按照以下列表中给出的模型名称填写（大小写敏感）。
2. 命令行参数"--quant_type"支持：w4a8, w4a16, w8a8, w8a8s, w8a8c8, w8a16。

### Qwen系列
| 模型名称                      |支持的量化类型|
|---------------------------|----|
| Qwen2.5-7B-Instruct       |w8a8|
| Qwen2.5-32B-Instruct      |w8a8|
| Qwen2.5-72B-Instruct      |w8a8c8|
| Qwen2.5-Coder-7B-Instruct |w8a8s|
| Qwen-QwQ-32B              |w8a8, w8a8s|
| Qwen3-32B                 |w8a8|
| Qwen3-14B                 |w8a8|

### LLaMa系列
（待更新）

### DeepSeek系列
（待更新）
