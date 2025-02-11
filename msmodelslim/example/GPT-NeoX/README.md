# GPT-NeoX 量化案例

## 模型介绍
- [GPT-NeoX-20B](https://github.com/EleutherAI/gpt-neox)是一个 200 亿参数的自回归语言模型，在 Pile 数据集上训练。它的架构类似于 GPT-3，并且与 GPT-J-6B 的架构几乎相同。其训练数据集包含大量英语文本，反映了该模型的通用性质。

#### GPT-Neox模型当前已验证的量化方法
- W8A8量化：GPT-NeoX-20B
 
#### 此模型仓已适配的模型版本
- [GPT-NeoX-20B](https://huggingface.co/EleutherAI/gpt-neox-20b)

## 环境配置

- 环境配置请参考[使用说明](https://gitee.com/ascend/msit/blob/master/msmodelslim/README.md)

## 量化权重生成

- 量化权重统一使用[quant_gpt_neox.py](./quant_gpt_neox.py)脚本生成，以下提供GPT-NeoX模型量化权重生成快速启动命令。

#### 量化参数说明
| 参数名 | 含义 | 默认值 | 使用方法 | 
| ------ | ---- | --- | -------- | 
| model_path | 浮点权重路径 | 无默认值 | 必选参数；<br>输入GPT-NeoX权重目录路径。 |
| save_directory | 量化权重路径 | 无默认值 | 必选参数；<br>输出量化结果目录路径。 |
| w_bit | 权重量化bit | 8 | 大模型量化场景下，可配置为8或16； <br>大模型稀疏量化场景下，需配置为4。 |
| a_bit | 激活值量化bit | 8 |大模型量化场景下，可配置为8或16； <br>大模型稀疏量化场景下，需配置为8。 |
| device_type | device类型 | npu | 可选值：['cpu', 'npu'] |
| calib_file | 量化校准数据 | boolq.jsonl | 存放校准数据的json文件。 |
| act_method | 激活值量化方法 | 3 |(1) 1代表Label-Free场景的min-max量化方式。 <br>(2) 2代表Label-Free场景的histogram量化方式。 <br>(3) 3代表Label-Free场景的自动混合min-max和histogram量化方式，LLM大模型场景下推荐使用。|
| anti_method | 离群值抑制参数 | 无默认值 |'m1': SmoothQuant算法。<br>'m2': SmoothQuant加强版算法，推荐使用。<br>'m3': AWQ算法。<br>'m4': smooth优化算法 。<br>'m5': CBQ量化算法。|
| disable_names | 手动回退的量化层名称 | 无默认值 | 用户可根据精度要求手动设置。 |
| disable_level | L自动回退等级 | L0 | 配置示例如下：<br>'L0'：默认值，不执行回退。<br>'L1'：回退1层。<br>'L2'：回退2层。<br>'L3'：回退3层。<br>'L4'：回退4层。<br>'L5'：回退5层。|


- 更多参数配置要求，请参考量化过程中配置的参数 [QuantConfig](https://gitee.com/ascend/msit/blob/dev/msmodelslim/docs/Python-API接口说明/大模型压缩接口/大模型量化接口/PyTorch/QuantConfig.md)
  以及量化参数配置类 [Calibrator](https://gitee.com/ascend/msit/blob/dev/msmodelslim/docs/Python-API接口说明/大模型压缩接口/大模型量化接口/PyTorch/Calibrator.md)


### 使用案例
- 请将{浮点权重路径}和{量化权重路径}替换为用户实际路径。
- 如果需要使用npu多卡量化，请先配置环境变量，支持多卡量化：
  ```shell
  export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
  ```

#### GPT-NeoX-20B模型量化
##### GPT-NeoX-20B w8a8量化
- 生成GPT-NeoX-20B模型w8a8量化权重，使用自动混合min-max和histogram的激活值量化方式，SmoothQuant加强版算法，自动回退前34层，在NPU上进行运算
  ```shell
  python3 quant_gpt_neox.py --model_path {浮点权重路径} --save_directory {W8A8量化权重路径} --w_bit 8 --a_bit 8 --device_type npu --anti_method m2  --disable_level L34
  ```
