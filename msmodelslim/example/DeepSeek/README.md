# DeepSeek 量化案例

## 模型介绍
- [DeepSeek-LLM](https://github.com/deepseek-ai/deepseek-LLM)从包含2T token的中英文混合数据集中，训练得到7B Base、7B Chat、67B Base与67B Chat四种模型

- [DeepSeek-V2](https://github.com/deepseek-ai/DeepSeek-V2)推出了MLA (Multi-head Latent Attention)，其利用低秩键值联合压缩来消除推理时键值缓存的瓶颈，从而支持高效推理；在FFN部分采用了DeepSeekMoE架构，能够以更低的成本训练更强的模型。

#### DeepSeek模型当前已验证的量化方法
- W8A8量化：DeepSeek-V2-Lite-Chat-16B, DeepSeek-V2-Chat-236B
- W8A16量化：DeepSeek-V2-Lite-Chat-16B, DeepSeek-V2-Chat-236B
 
#### 此模型仓已适配的模型版本
- [Deepseek-V2-Chat](https://huggingface.co/deepseek-ai/DeepSeek-V2-Chat)

## 环境配置

- 环境配置请参考[使用说明](https://gitee.com/ascend/msit/blob/master/msmodelslim/README.md)

## 量化权重生成

- 量化权重统一使用[quant_deepseek.py](./quant_deepseek.py)脚本生成，以下提供DeepSeek模型量化权重生成快速启动命令。

#### 量化参数说明
| 参数名 | 含义 | 使用方法 | 
| ------ | ---- | -------- | 
| model_path | 浮点权重路径 | 输入LLAMA权重目录路径 |
| save_directory | 量化权重路径 | 输出量化结果目录路径 |
| a_bit | 激活值量化bit | 大模型量化场景下，可配置为8或16 <br>大模型稀疏量化场景下，需配置为8 |
| w_bit | 权重量化bit | 大模型量化场景下，可配置为8或16 <br>大模型稀疏量化场景下，需配置为4 |
| device_type | device类型 | 可选值：['cpu', 'npu']，默认为'cpu' |
| disable_names | 手动回退的量化层名称 | 用户可根据精度要求手动设置，默认回退隐藏层的降维投影层 |
| disable_level | L自动回退等级 | 配置示例如下：<br>'L0':默认值，不执行回退。<br>'L1'：回退1层。<br>'L2'：回退2层。<br>'L3'：回退3层。<br>'L4'：回退4层。<br>'L5'：回退5层。|
| act_method | 激活值量化方法 | 可选值如下所示，默认为1。<br>(1) 1代表Label-Free场景的min-max量化方式。 <br>(2) 2代表Label-Free场景的histogram量化方式。 <br>(3) 3代表Label-Free场景的自动混合量化方式，LLM大模型场景下推荐使用。|
| part_file_size | 单个权重文件最大大小 | DeepSeek场景下默认值为5GB |

- 更多参数配置要求，请参考量化过程中配置的参数 [QuantConfig](https://gitee.com/ascend/msit/blob/dev/msmodelslim/docs/Python-API接口说明/大模型压缩接口/大模型量化接口/PyTorch/QuantConfig.md)
  以及量化参数配置类 [Calibrator](https://gitee.com/ascend/msit/blob/dev/msmodelslim/docs/Python-API接口说明/大模型压缩接口/大模型量化接口/PyTorch/Calibrator.md)


### 使用案例
- 请将{浮点权重路径}和{量化权重路径}替换为用户实际路径。
- 如果需要使用npu多卡量化，请先配置环境变量，支持多卡量化：
  ```shell
  export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
  ```

#### DeepSeek-V2模型量化
##### DeepSeek-V2 w8a16量化
- 生成DeepSeek-V2模型w8a16量化权重，使用histogram量化方式，在CPU上进行运算
  ```shell
  python3 quant_deepseek.py --model_path {浮点权重路径} --save_directory {W8A16量化权重路径} --device_type cpu --act_method 2 --w_bit 8 --a_bit 16
  ```

##### DeepSeek-V2 w8a8 dynamic量化
- 生成DeepSeek-V2模型w8a8 dynamic量化权重，使用histogram量化方式，在CPU上进行运算
  ```shell
  python3 quant_deepseek.py --model_path {浮点权重路径} --save_directory {W8A8量化权重路径} --device_type cpu --act_method 2 --w_bit 8 --a_bit 8  --is_dynamic True
  ```
