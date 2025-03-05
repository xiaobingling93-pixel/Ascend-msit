# LLAMA 量化案例

## 模型介绍

- [LLaMA（Large Language Model Meta AI）](https://github.com/facebookresearch/llama/tree/llama_v1)和 [LLaMA2（Large Language Model Meta AI 2）](https://github.com/facebookresearch/llama)，是由 Meta AI 发布的一个开放且高效的大型基础语言模型，可以通过自然语言交互的方式提供知识、文本生成、语言翻译、语言理解、代码编写和解释等任务。

#### LLAMA模型当前已验证的量化方法

- W8A8量化：LLaMa2-7B，LLaMa2-13B，LLaMa2-70B, Llama3.1-8B, Llama3.1-70B
- W8A16量化：LLaMa-65B，LLaMa2-70B, Llama3-70B
- 稀疏量化：LLaMa-33B，LLaMa2-7B，LLaMa2-13B, Llama3.1-70B
- KV cache量化: Llama3.1-70B
- Attention 量化：Llama3.1-70B

#### 此模型仓已适配的模型版本权重获取地址
  - [LLaMa系列](https://github.com/facebookresearch/llama/tree/llama_v1)
  - [LLaMa2系列](https://github.com/facebookresearch/llama/tree/v2)
  - [LLaMa3系列](https://github.com/meta-llama/llama3)

## 环境配置

- 环境配置请参考[使用说明](https://gitee.com/ascend/msit/blob/master/msmodelslim/README.md)

## 量化权重生成

- 量化权重统一使用[quant_llama.py](./quant_llama.py)脚本生成，以下提供LLaMa模型量化权重生成快速启动命令

#### 量化参数说明

| 参数名 | 含义 | 默认值 | 使用方法 | 
| ------ | ---- | --- | -------- | 
| model_path | 浮点权重路径 | 无默认值 | 必选参数；<br>输入Llama权重目录路径。 |
| model_type | qwen模型类型 | llama2|  若使用llama3模型，请输入该参数为llama3。 |
| save_directory | 量化权重路径 | 无默认值 | 必选参数；<br>输出量化结果目录路径。 |
| a_bit | 激活值量化bit | 8 |大模型量化场景下，可配置为8或16； <br>大模型稀疏量化场景下，需配置为8。 |
| w_bit | 权重量化bit | 8 | 大模型量化场景下，可配置为8或16； <br>大模型稀疏量化场景下，需配置为4。 |
| device_type | device类型 | cpu | 可选值：['cpu', 'npu'] |
| calib_file | 量化校准数据 | teacher_qualification.jsonl | 存放校准数据的json文件。 |
| disable_names | 手动回退的量化层名称 | w8a8量化默认回退所有down_proj层 <br> Llama3 w8a16量化默认回退前5层 | 用户可根据精度要求手动设置，默认回退隐藏层的降维投影层。 |
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
| do_smooth | 是否开启smooth功能 | False | 是否开启smooth quant算法。|
| anti_calib_file | 离群值抑制校准数据文件 | None | 用于离群值抑制的校准数据文件路径(.json或.jsonl格式)。|
| disable_threshold | 自动选择回退层的阈值 | 0 | 当值大于0时，会根据该阈值自动选择需要回退的层。值越大，回退的层越多。|
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

#### LLaMa
##### LLaMa-65B W8A16量化
  ```shell
  python3 quant_llama.py --model_path {浮点权重路径} --save_directory {W8A16量化权重路径} --calib_file ../common/teacher_qualification.jsonl --w_bit 8 --a_bit 16 --act_method 3 
  ```
##### LLaMa 33B 稀疏量化配置
  ```shell
  python3 quant_llama.py --model_path {浮点权重路径} --save_directory {W8A8S量化权重路径} --calib_file ../common/boolq.jsonl --act_method 2 --do_smooth True --use_sigma True --is_lowbit True --co_sparse True --w_bit 4 
  ```
#### LLaMa2
##### LLaMa2-7B/13B W8A8量化
- 生成llama2-7b量化权重，antioutlier使用m1算法配置，使用min-max量化方式，校准数据集使用50条BoolQ数据，在CPU上进行运算
  ```shell
  python3 quant_llama.py --model_path {浮点权重路径} --save_directory {W8A8量化权重路径} --calib_file ../common/boolq.jsonl --device_type cpu --disable_level L0 --anti_method m1 --act_method 1 
  ```

- 生成llama2-13b量化权重，antioutlier使用m2算法配置，使用min-max量化方式，校准数据集使用50条BoolQ数据，在CPU上进行运算
  ```shell
  python3 quant_llama.py --model_path {浮点权重路径} --save_directory {W8A8量化权重路径} --calib_file ../common/boolq.jsonl  --device_type cpu --disable_level L0 --anti_method m2 --act_method 1 
  ```

##### LLaMa2 7B/13B 稀疏量化配置
  ```shell
  python3 quant_llama.py --model_path {浮点权重路径} --save_directory {W4A8S量化权重路径} --calib_file ../common/teacher_qualification.jsonl --w_bit 4 --a_bit 8 --fraction 0.011 --co_sparse True 
  ```

##### LLaMa2-70B NPU多卡W8A8量化
  ```shell
  python3 quant_llama.py --model_path {浮点权重路径} --save_directory {W8A8量化权重路径} --calib_file ../common/boolq.jsonl  --device_type npu --disable_level L5 
  ```


##### LLaMa2-70B W8A16量化
  ```shell
  python3 quant_llama.py --model_path {浮点权重路径} --save_directory {W8A16量化权重路径} --calib_file= ../common/teacher_qualification.jsonl --w_bit 8 --a_bit 16 --act_method 3 
  ```

#### LLaMa3
##### LLaMa3-70B W8A16量化
  ```shell
  python3 quant_llama.py --model_path {浮点权重路径} --save_directory {W8A16量化权重路径} --calib_file ../common/boolq.jsonl  --device_type npu --a_bit 16 --w_sym False --mm_tensor False --anti_method m3 --act_method 3 --model_type llama3
  ```
#### LLaMa3.1
##### Llama3.1-8B W8A8量化：
  ```shell
  python3 quant_llama.py --model_path {浮点权重路径} --save_directory {W8A8量化权重路径} --calib_file ../common/boolq.jsonl  --device_type npu --disable_level L0 --anti_method m1 --act_method 1 
  ```
##### Llama3.1-70B W8A8量化：
  ```shell
  python3 quant_llama.py --model_path {浮点权重路径} --save_directory {W8A8量化权重路径} --calib_file ../common/boolq.jsonl  --device_type npu --disable_level L5 --anti_method m3 --act_method 3 
  ```

##### Llama3.1-70B W8A8量化搭配 KV cache int8量化
  ```shell
  python3 quant_llama.py --model_path {浮点权重路径} --save_directory {W8A8量化权重路径} --calib_file ../common/boolq.jsonl  --device_type npu --disable_level L5 --anti_method m3 --act_method 3 --use_kvcache_quant True
  ```
##### Llama3.1-70B W8A8量化搭配Attention量化
- 当前仅支持基于BF16权重生成量化权重
- 需修改`modeling_llama.py`文件和`config.json`文件，配置方法参考[FA量化使用说明](https://gitee.com/ascend/msit/blob/master/msmodelslim/docs/FA%E9%87%8F%E5%8C%96%E4%BD%BF%E7%94%A8%E8%AF%B4%E6%98%8E.md)。 
- 相比于W8A8量化，需额外设置`use_fa_quant`参数为True
  ```shell
  python3 quant_llama.py --model_path {浮点权重路径} --save_directory {W8A8量化权重路径} --calib_file ../common/boolq.jsonl --w_bit 8 --a_bit 8 --device_type npu --disable_level L5 --anti_method m4 --act_method 3 --use_fa_quant True
  ```
##### Llama3.1-70B W8A8-pdmix量化(prefill阶段 w8a8动态量化, decode阶段 w8a8量化) 搭配 KV cache int8量化
  ```shell
  python3 quant_llama.py --model_path {浮点权重路径} \
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