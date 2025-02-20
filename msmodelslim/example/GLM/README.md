
# GLM 量化案例

## 模型介绍
- [GLM](https://github.com/THUDM/GLM)是智谱AI推出的最新一代预训练模型GLM-4系列中的开源版本。在语义、数学、推理、代码和知识等多方面的数据集测评中，GLM-4-9B 及其人类偏好对齐的版本GLM-4-9B-Chat均表现出超越 Llama-3-8B的卓越性能。除了能进行多轮对话，GLM-4-9B-Chat还具备网页浏览、代码执行、自定义工具调用（Function Call）和长文本推理（支持最大128K上下文等高级功能。本代模型增加了多语言支持，支持包括日语，韩语，德语在内的26种语言。我们还推出了支持 1M 上下文长度（约 200 万中文字符）的GLM-4-9B-Chat-1M模型和基于 GLM-4-9B的多模态模型GLM-4V-9B。GLM-4V-9B具备1120 * 1120高分辨率下的中英双语多轮对话能力，在中英文综合能力、感知推理、文字识别、图表理解等多方面多模态评测中，GLM-4V-9B表现出超越GPT-4-turbo-2024-04-09、Gemini 1.0 Pro、Qwen-VL-Max 和 Claude 3 Opus的卓越性能。

#### GLM模型当前已验证的量化方法
- W8A8C8量化：GLM-4-9B
 
#### 此模型仓已适配的模型版本
- [GLM-4-9B](https://huggingface.co/THUDM/glm-4-9b)

## 环境配置

- 环境配置请参考[使用说明](https://gitee.com/ascend/msit/blob/master/msmodelslim/README.md)

## 量化权重生成

- 量化权重统一使用[quant_glm.py](./quant_glm.py)脚本生成，以下提供GLM模型量化权重生成快速启动命令。

#### 量化参数说明
| 参数名               | 含义                   | 默认值 | 使用方法                                                                                                                    | 
|-------------------|----------------------| --- |-------------------------------------------------------------------------------------------------------------------------| 
| model_path        | 浮点权重路径               | 无默认值 | 必选参数；<br>输入DeepSeek权重目录路径。                                                                                              |
| save_directory    | 量化权重路径               | 无默认值 | 必选参数；<br>输出量化结果目录路径。                                                                                                    |
| a_bit             | 激活值量化bit             | 8 | 大模型量化场景下，可配置为8或16； <br>大模型稀疏量化场景下，需配置为8。                                                                                |
| w_bit             | 权重量化bit              | 8 | 大模型量化场景下，可配置为8或16； <br>大模型稀疏量化场景下，需配置为4。                                                                                |
| device_type       | device类型             | cpu | 可选值：['cpu', 'npu']                                                                                                      |
| calib_file        | 量化校准数据               | mix_dataset_glm.json | 存放量化校准数据的json文件。                                                                                                        |
| anti_file         | 异常值抑制校准数据            | mix_dataset_glm.json | 存放异常值抑制校准数据的json文件。                                                                                                     |
| disable_names     | 手动回退的量化层名称           | 默认回退所有down_proj层 | 用户可根据精度要求手动设置，默认回退隐藏层的降维投影层。                                                                                            |
| disable_level     | L自动回退等级              | L0 | 配置示例如下：<br>'L0'：默认值，不执行回退。<br>'L1'：回退1层。<br>'L2'：回退2层。<br>'L3'：回退3层。<br>'L4'：回退4层。<br>'L5'：回退5层。                        |
| act_method        | 激活值量化方法              | 1 | (1) 1代表Label-Free场景的min-max量化方式。 <br>(2) 2代表Label-Free场景的histogram量化方式。 <br>(3) 3代表Label-Free场景的自动混合量化方式，LLM大模型场景下推荐使用。 |
| anti_method       | 离群值抑制参数              | 无默认值 | 'm1': SmoothQuant算法。<br>'m2': SmoothQuant加强版算法，推荐使用。<br>'m3': AWQ算法。<br>'m4': smooth优化算法 。<br>'m5': CBQ量化算法。<br>默认为m2。  |
| co_sparse	        | 是否开启稀疏量化功能           | False | True: 使用稀疏量化功能；<br>False: 不使用稀疏量化功能。                                                                                    |
| fraction          | 模型权重稀疏量化过程中被保护的异常值占比 |0.01| 取值范围[0.01,0.1]                                                                                                          |
| use_sigma         | 是否启动sigma功能          | False| True: 开启sigma功能；<br>False: 不开启sigma功能。                                                                                  |
| is_lowbit         | 是否开启lowbit量化功能       | False| (1) 当w_bit=4，a_bit=8时，为大模型稀疏量化场景，表示开启lowbit稀疏量化功能。<br>(2) 其他场景为大模型量化场景，会开启量化自动精度调优功能。<br>当前量化自动精度调优框架支持W8A8，W8A16量化。    |
| part_file_size    | 量化权重文件大小             | 无限制 | 单个量化权重文件大小不超过xGB。                                                                                                       |
| use_kvcache_quant | 是否使用kvcache量化功能      | False | True: 使用kvcache量化功能；<br>False: 不使用kvcache量化功能。                                                                          |
| is_dynamic        | 是否使用per-token动态量化功能  | False | True: 使用per-token动态量化；<br>False: 不使用per-token动态量化。                                                                      |


- 更多参数配置要求，请参考量化过程中配置的参数 [QuantConfig](https://gitee.com/ascend/msit/blob/dev/msmodelslim/docs/Python-API接口说明/大模型压缩接口/大模型量化接口/PyTorch/QuantConfig.md)
  以及量化参数配置类 [Calibrator](https://gitee.com/ascend/msit/blob/dev/msmodelslim/docs/Python-API接口说明/大模型压缩接口/大模型量化接口/PyTorch/Calibrator.md)


### 使用案例
- 请将{浮点权重路径}和{量化权重路径}替换为用户实际路径。
- 如果需要使用npu多卡量化，请先配置环境变量，支持多卡量化，但GLM-4-9B量化仅需要单卡：
  ```shell
  export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
  ```

#### GLM-4-9B模型量化
##### GLM-4-9B w8a8c8量化
- 生成GLM-4-9B模型w8a8c8量化权重，使用histogram量化方式，在NPU上进行运算
  ```shell
  python3 quant_glm.py --model_path {浮点权重路径} --save_directory {W8A8C8量化权重路径} --device_type npu --act_method 2 --anti_method m4 --disable_level L0 --w_bit 8 --a_bit 8 --use_kvcache_quant True --calib_file ../common/mix_dataset_glm.json --anti_file ../common/mix_dataset_glm.json
  ```