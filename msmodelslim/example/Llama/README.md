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

## 环境配置

- 环境配置请参考[使用说明](https://gitee.com/ascend/msit/blob/master/msmodelslim/README.md)

## 量化权重生成

- 量化权重统一使用[quant_llama.py](./quant_llama.py)脚本生成，以下提供LLaMa模型量化权重生成快速启动命令

#### 量化参数说明

| 参数名 | 含义 | 使用方法 | 
| ------ | ---- | -------- | 
| model_path | 浮点权重路径 | 输入LLAMA权重目录路径 |
| save_directory | 量化权重路径 | 输出量化结果目录路径 |
| a_bit | 激活值量化bit | 大模型量化场景下，可配置为8或16 <br>大模型稀疏量化场景下，需配置为8 |
| w_bit | 权重量化bit | 大模型量化场景下，可配置为8或16 <br>大模型稀疏量化场景下，需配置为4 |
| device_type | device类型 | 可选值：['cpu', 'npu']，默认为'cpu' |
| calib_file | 量化校准数据 | 存放校准数据的json文件 |
| disable_names | 手动回退的量化层名称 | 用户可根据精度要求手动设置，默认回退隐藏层的降维投影层 |
| disable_level | L自动回退等级 | 配置示例如下：<br>'L0':默认值，不执行回退。<br>'L1'：回退1层。<br>'L2'：回退2层。<br>'L3'：回退3层。<br>'L4'：回退4层。<br>'L5'：回退5层。|
| w_sym	| 权重量化是否为对称量化 | 默认为True，W8A8场景仅支持配置为True|
| act_method | 激活值量化方法 | 可选值如下所示，默认为1。<br>(1) 1代表Label-Free场景的min-max量化方式。 <br>(2) 2代表Label-Free场景的histogram量化方式。 <br>(3) 3代表Label-Free场景的自动混合量化方式，LLM大模型场景下推荐使用。|
| anti_method | 离群值抑制参数 |'m1': SmoothQuant算法。<br>'m2': SmoothQuant加强版算法，推荐使用。<br>'m3': AWQ算法。<br>'m4': smooth优化算法 。<br>'m5': CBQ量化算法。<br>默认为m2。|
| co_sparse	| 是否开启稀疏量化功能 | 默认值：False，不开启稀疏量化 |
| fraction | 模型权重稀疏量化过程中被保护的异常值占比  | 取值范围[0.01,0.1]。默认值为0.01。|


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


##### LLaMa1 33B 稀疏量化配置
  ```shell
  python3 quant_llama.py --model_path {浮点权重路径} --save_directory {W8A8S量化权重路径} --calib_file ../common/boolq.jsonl --act_method 2 --do_smooth True --use_sigma True --is_lowbit True --co_sparse True --w_bit 4 
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
  python3 quant_llama.py --model_path {浮点权重路径} --save_directory {W8A8量化权重路径} --calib_file ../common/boolq.jsonl  --device_type npu --disable_level L5 --anti_method m3 --act_method 3 --use_fa_quant True
  ```