# InternVL 2.0 量化案例

## 模型介绍

- [InternVL 2.0](https://internvl.github.io/blog/2024-07-02-InternVL-2.0/) 是上海人工智能实验室联合商汤科技开发的书生·万象多模态大模型。InternVL 2.0的关键评测指标比肩国际顶尖商用闭源模型，支持图像、视频、文本、语音、三维、医疗多种模态支持百种下游任务，性能媲美任务专用模型。书生·万象在处理复杂多模态数据方面具有强大能力，尤其是在数学、科学图表、通用图表、文档、信息图表和OCR等任务中表现优异。

#### InternVL 2.0模型当前已验证的量化方法

- W8A8量化：InternVL 2.0

#### 此模型仓已适配的模型版本权重获取地址
##### InternVL 2.0
- [InternVL 2.0](https://huggingface.co/spaces/OpenGVLab/InternVL)

## 环境配置

- 环境配置请参考[使用说明](https://gitee.com/ascend/msit/blob/master/msmodelslim/README.md)
- transformers版本请参照模型路径下config.json配置为4.37.2

## 量化权重生成

- 量化权重统一使用[quant_internvl2.py](./quant_qwen2vl.py)脚本生成，以下提供Qwen2VL模型量化权重生成快速启动命令。


#### 量化参数说明
| 参数名 | 含义 | 默认值 | 使用方法 | 
| ------ | ---- | --- | -------- | 
| model_path | 浮点权重路径 | 无默认值 | 必选参数；<br>输入InternVL 2.0权重目录路径。 |
| calib_images | 校准集图片路径 | ./textvqa_val | 可选参数；<br>输入校准数据集的目录路径。本示例中图片来源于公开数据集[textvqa](https://huggingface.co/datasets/maoxx241/textvqa_subset)。 |
| save_directory | 量化权重路径 | 无默认值 | 必选参数；<br>输出量化结果目录路径。 |
| a_bit | 激活值量化bit | 8 |大模型量化场景下，可配置为8或16； <br>大模型稀疏量化场景下，需配置为8。 <br>InternVL 2.0当前仅支持配置为8。|
| w_bit | 权重量化bit | 8 | 大模型量化场景下，可配置为8或16； <br>大模型稀疏量化场景下，需配置为4。 <br>InternVL 2.0当前仅支持配置为8。|
| device_type | device类型 | cpu | 可选值：['cpu', 'npu'] |
| part_file_size | 量化权重文件大小 | 无限制 | 单个量化权重文件大小不超过xGB。|

- 更多参数配置要求，请参考量化过程中配置的参数 [QuantConfig](https://gitee.com/ascend/msit/blob/dev/msmodelslim/docs/Python-API接口说明/大模型压缩接口/大模型量化接口/PyTorch/QuantConfig.md)
  以及量化参数配置类 [Calibrator](https://gitee.com/ascend/msit/blob/dev/msmodelslim/docs/Python-API接口说明/大模型压缩接口/大模型量化接口/PyTorch/Calibrator.md)

### 使用案例
- 请将{浮点权重路径}和{量化权重路径}替换为用户实际路径。
- 如果需要使用npu多卡量化(特别是InternVL 2.0-40B这种大模型)，请先配置环境变量，暂不支持Atlas推理系列产品，以下npu 8卡量化为例：
  ```shell
  export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
  ```
  
#### 1. InternVL 2.0系列
##### InternVL 2.0-8B W8A8量化
生成InternVL 2.0-8B模型量化权重，AntiOutlier异常值抑制使用m2算法配置（当前仅支持m2），在NPU上进行运算。
  ```shell
  python quant_internvl2.py  --model_path {浮点权重路径} --calib_images {校准图片路径}  --save_directory {量化权重保存路径} --w_bit 8 --a_bit 8 --device_type npu --is_8B_model True
  ```

##### InternVL 2.0-40B W8A8量化
生成InternVL 2.0-40B模型量化权重，AntiOutlier异常值抑制使用m2算法配置（当前仅支持m2），在NPU上进行运算。
  ```shell
  python quant_internvl2.py  --model_path {浮点权重路径} --calib_images {校准图片路径}  --save_directory {量化权重保存路径} --w_bit 8 --a_bit 8 --device_type npu --is_8B_model False
  ```
