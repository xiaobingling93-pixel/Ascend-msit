# Qwen 量化案例

## 模型介绍
- 千问（qwen）语言大模型是阿里巴巴集团推出的大型语言模型，具备强大的自然语言处理能力，能够理解和生成文本，应用于智能客服、内容生成、问答系统等多个场景，助力企业智能化升级。

#### Qwen模型当前已验证的量化方法
- W8A8量化：Qwen-7B，Qwen-14B，Qwen1.5-14B，Qwen1.5-32B，Qwen2-7B，Qwen2-72B，Qwen2.5-7B，Qwen2.5-14B，Qwen2.5-32B
- W8A16量化：QWen-72B，Qwen1.5-72B，Qwen1.5-110B，Qwen2-72B
- 稀疏量化：Qwen1.5-14B，Qwen2-7B，Qwen2-72B，Qwen2.5-7B，Qwen2.5-14B，QwenCode2.5-7B
- KV cache量化：Qwen2-72B
- Attention 量化：Qwen2.5-72B

#### 此模型仓已适配的模型版本
##### Qwen
- [Qwen-7B](https://huggingface.co/Qwen/Qwen-7B/tree/main)
- [Qwen-14B](https://huggingface.co/Qwen/Qwen-14B/tree/main)
- [QWen-72B](https://huggingface.co/Qwen/Qwen-72B/tree/main)
##### Qwen1.5
- [Qwen1.5-14B](https://huggingface.co/Qwen/Qwen1.5-14B/tree/main)
- [Qwen1.5-32B](https://huggingface.co/Qwen/Qwen1.5-32B/tree/main)
- [Qwen1.5-72B](https://huggingface.co/Qwen/Qwen1.5-72B/tree/main)
##### Qwen2
- [Qwen2-7B](https://huggingface.co/Qwen/Qwen2-7B/tree/main)
- [Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct/tree/main)
##### Qwen2.5
- [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/tree/main)
- [Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct/tree/main)
- [Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/tree/main)
- [Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct/tree/main)

## 环境配置

- 环境配置请参考[使用说明](https://gitee.com/ascend/msit/blob/master/msmodelslim/README.md)

## 量化权重生成

- 量化权重统一使用[quant_qwen.py.py](./quant_qwen.py)脚本生成，以下提供Qwen模型量化权重生成快速启动命令。


#### 量化参数说明
| 参数名 | 含义 | 默认值 | 使用方法 | 
| ------ | ---- | --- | -------- | 
| model_path | 浮点权重路径 | 无默认值 | 必选参数；<br>输入Qwen权重目录路径。 |
| model_type | qwen模型类型 | qwen2|  若使用qwen1模型，请输入该参数为qwen1。 |
| save_directory | 量化权重路径 | 无默认值 | 必选参数；<br>输出量化结果目录路径。 |
| a_bit | 激活值量化bit | 8 |大模型量化场景下，可配置为8或16； <br>大模型稀疏量化场景下，需配置为8。 |
| w_bit | 权重量化bit | 8 | 大模型量化场景下，可配置为8或16； <br>大模型稀疏量化场景下，需配置为4。 |
| device_type | device类型 | cpu | 可选值：['cpu', 'npu'] |
| calib_file | 量化校准数据 | teacher_qualification.jsonl | 存放校准数据的json文件。 |
| disable_names | 手动回退的量化层名称 | qwen1 回退所有c_proj层 <br> 其他模型默认回退所有down_proj层 | 用户可根据精度要求手动设置，默认回退隐藏层的降维投影层。 |
| disable_level | L自动回退等级 | L0 | 配置示例如下：<br>'L0'：默认值，不执行回退。<br>'L1'：回退1层。<br>'L2'：回退2层。<br>'L3'：回退3层。<br>'L4'：回退4层。<br>'L5'：回退5层。|
| act_method | 激活值量化方法 | 1 |(1) 1代表Label-Free场景的min-max量化方式。 <br>(2) 2代表Label-Free场景的histogram量化方式。 <br>(3) 3代表Label-Free场景的自动混合量化方式，LLM大模型场景下推荐使用。|
| anti_method | 离群值抑制参数 | 无默认值 |'m1': SmoothQuant算法。<br>'m2': SmoothQuant加强版算法。<br>'m3': AWQ算法。<br>'m4': smooth优化算法 。<br>'m5': CBQ量化算法。<br>'m6': Flex smooth量化算法。|
| co_sparse	| 是否开启稀疏量化功能 | False | True: 使用稀疏量化功能；<br>False: 不使用稀疏量化功能。 |
| fraction | 模型权重稀疏量化过程中被保护的异常值占比  |0.01| 取值范围[0.01,0.1]|
| use_sigma | 是否启动sigma功能 | False|True: 开启sigma功能；<br>False: 不开启sigma功能。 |
| is_lowbit | 是否开启lowbit量化功能 | False|(1) 当w_bit=4，a_bit=8时，为大模型稀疏量化场景，表示开启lowbit稀疏量化功能。<br>(2) 其他场景为大模型量化场景，会开启量化自动精度调优功能。<br>当前量化自动精度调优框架支持W8A8，W8A16量化。|
| part_file_size | 量化权重文件大小 | 无限制 | 单个量化权重文件大小不超过xGB。|
| use_kvcache_quant | 是否使用kvcache量化功能 | False | True: 使用kvcache量化功能；<br>False: 不使用kvcache量化功能。|
| is_dynamic | 是否使用per-token动态量化功能 | False | True: 使用per-token动态量化；<br>False: 不使用per-token动态量化。 |
| disable_threshold | 自动回退阈值 | 0 | 当值大于0时，会根据阈值自动选择需要回退的层。|
| anti_calib_file | 离群值抑制校准数据文件 | None | 用于离群值抑制的校准数据文件路径(.json或.jsonl)。|
| pdmix | 是否使用PDMix量化类型 | False | True: 使用PDMix量化类型；<br>False: 不使用PDMix量化类型。|

- 更多参数配置要求，请参考量化过程中配置的参数 [QuantConfig](https://gitee.com/ascend/msit/blob/dev/msmodelslim/docs/Python-API接口说明/大模型压缩接口/大模型量化接口/PyTorch/QuantConfig.md)
  以及量化参数配置类 [Calibrator](https://gitee.com/ascend/msit/blob/dev/msmodelslim/docs/Python-API接口说明/大模型压缩接口/大模型量化接口/PyTorch/Calibrator.md)

### 使用案例
- 请将{浮点权重路径}和{量化权重路径}替换为用户实际路径。
- 如果需要使用npu多卡量化，请先配置环境变量，支持多卡量化：
  ```shell
  export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
  ```
  
#### 1. Qwen1系列
##### Qwen1-14b W8A8量化
生成Qwen1-14b模型量化权重，antioutlier使用m2算法配置，使用min-max量化方式，在CPU上进行运算
  ```shell
  python3 quant_qwen.py --model_path {浮点权重路径} --save_directory {W8A8量化权重路径} --calib_file ../common/boolq.jsonl --w_bit 8 --a_bit 8 --device_type cpu  --anti_method m2 --act_method 1 --model_type qwen1
  ```

##### Qwen1-72b W8A16量化
生成Qwen1-14b模型量化权重，激活值量化使用自动混合量化方式，在CPU上进行运算
  ```shell
  python3 quant_qwen.py --model_path {浮点权重路径} --save_directory {W8A16量化权重路径} --calib_file ../common/ceval.jsonl --w_bit 8 --a_bit 16 --device_type cpu  --act_method 3 --model_type qwen1
  ```
#### 2. Qwen1.5系列
##### Qwen1.5-14b, Qwen1.5-32b W8A8 量化
生成量化权重，使用min-max量化方式，校准数据集使用50条BoolQ数据，在npu上进行运算
  ```shell
  python3 quant_qwen.py --model_path {浮点权重路径} --save_directory {W8A8量化权重路径} --calib_file ../common/boolq.jsonl --w_bit 8 --a_bit 8 --device_type npu 
  ```
 稀疏量化权重请使用以下指令生成
  ```shell
  python3 quant_qwen.py --model_path {浮点权重路径} --save_directory {W8A8量化权重路径} --calib_file ../common/cn_en.jsonl --w_bit 4 --a_bit 8 --device_type npu --fraction 0.011 --use_sigma True --is_lowbit True 
  ```
##### Qwen1.5-72b W8A16量化
  ```shell
  python3 quant_qwen.py --model_path {浮点权重路径} --save_directory {W8A16量化权重路径}  --w_bit 8 --a_bit 16 --device_type npu 
  ```
#### 3. Qwen2系列
##### Qwen2-7b W8A8量化
  ```shell
  python3 quant_qwen.py --model_path {浮点权重路径} --save_directory {W8A8量化权重路径} --calib_file ../common/boolq.jsonl --w_bit 8 --a_bit 8 --device_type npu 
  ```
##### Qwen2-72b W8A16量化
  ```shell
  python3 quant_qwen.py --model_path {浮点权重路径} --save_directory {W8A16量化权重路径}  --w_bit 8 --a_bit 16 --device_type npu 
  ```
稀疏量化权重请使用以下指令生成
  ```shell
  export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False

  python3 quant_qwen.py --model_path {浮点权重路径} --save_directory {W8A8s量化权重路径} --calib_file ../common/cn_en.jsonl --w_bit 4 --a_bit 8 --fraction 0.011 --co_sparse True --device_type npu --use_sigma True --is_lowbit True
  ```

##### Qwen2-72b KV Cache W8A8量化
  ```shell
  export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False

  python3 quant_qwen.py --model_path {浮点权重路径} --save_directory {W8A16量化权重路径} --calib_file ../common/boolq.jsonl --w_bit 8 --a_bit 8 --device_type npu   --use_kvcache_quant True
  ```

#### 4. Qwen2.5系列
##### Qwen2.5-7b, Qwen2.5-14b, Qwen2.5-32b W8A8 量化
  ```shell
  python3 quant_qwen.py --model_path {浮点权重路径} --save_directory {W8A8量化权重路径} --calib_file ../common/boolq.jsonl --w_bit 8 --a_bit 8 --device_type npu  
  ```
##### Qwen2.5-72B 支持Attention量化
- 需修改`modeling_llama.py`文件和`config.json`文件，配置方法参考[FA量化使用说明](https://gitee.com/ascend/msit/blob/master/msmodelslim/docs/FA%E9%87%8F%E5%8C%96%E4%BD%BF%E7%94%A8%E8%AF%B4%E6%98%8E.md)。 
- 相比于W8A8量化，需额外设置`use_fa_quant`参数为True
  ```shell
  python3 quant_qwen.py --model_path {浮点权重路径} --save_directory {W8A8量化权重路径} --calib_file ../common/boolq.jsonl --w_bit 8 --a_bit 8 --device_type npu --anti_method m4 --act_method 1 --use_fa_quant True
  ```
##### Qwen2.5-Coder-7B 稀疏量化 
  ```shell
  python3 quant_qwen.py --model_path {浮点权重路径} --save_directory {W4A8量化权重路径} --calib_file ../common/humaneval_x.jsonl --w_bit 4 --a_bit 8 --device_type cpu --fraction 0.02 --co_sparse True  --use_sigma True --is_lowbit False
  ```
##### Qwen2.5-72b W8A8-pdmix量化(prefill阶段 w8a8动态量化, decode阶段 w8a8量化) 搭配 KV cache int8量化
  ```shell
  python3 quant_qwen.py --model_path {浮点权重路径} \
  --save_directory {W8A8-pdmix量化权重路径} \
  --calib_file ./calib_data/calib_prompt.jsonl  \
  --anti_calib_file ./calib_data/anti_calib_prompt.jsonl \
  --device_type npu \
  --anti_method m6 \
  --act_method 3 \
  --use_kvcache_quant True \
  --disable_threshold 1 \
  --pdmix True
  ```