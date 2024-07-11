
# msmodelslim量化权重格式

msmodelslim llm-ptq工具生成的safetensors量化权重文件包含两个文件，quant_model_weight.safetensors权重文件和quant_model_description.json权重描述文件。

## W8A16, W4A16量化

量化工具对于每个量化的Linear生成三个参数，参数名称为：weight、weight_scale、weight_offset，在safetensors权重文件中，完整的权重名称为Lienar层的名称+参数名称，例如ChatGLM2-6B量化权重中，"transformer.encoder.layers.0.self_attention.query_key_value.weight_scale"，"transformer.encoder.layers.0.self_attention.query_key_value"为Linear层的名称，"weight_scale"为参数的名称。

weight为量化后的int8或int4的权重，数据类型为torch.Tensor，dtype为torch.int8，shape和原始浮点的shape一致，记为n, k = weight.shape。
weight_scale为量化的缩放系数，数据类型为torch.Tensor，dtype为torch.float16或torch.float32，在per_channel场景下，shape为[n, 1]，在per_group场景下，shape为[n, k / group_size]。
weight_offset为量化的偏移系数，数据类型为torch.Tensor，dtype和shape和weight_scale一致。对称量化场景下需要构造全0的weight_offset。

反量化的计算公式为:  
per_channel 场景：  
deq_weight = (weight - weight_offset) * weight_scale  
per_group 场景  
weight = weight.reshape((-1, group_size))  
weight_offset = weight_offset.reshape((n * k / group_size, 1))  
weight_scale = weight_scale.reshape((n * k / group_size, 1))  
deq_weight = ((weight - weight_offset) * weight_scale).reshape((n, k))

对于W4A16场景，因为原生pytorch不支持torch.int4的tensor类型，开源的模型、量化工具有各自的实现方式，例如ChatGLM2-6B采用两个int4的tensor合并成int8的tensor，0x1,0x2,0x3,0x4合并成0x12,0x34；AutoAWQ采用8个int4的tensor合并成int32的tensor，0x1,0x2...0x8合并成0x87654321，不能整除的部分补0。msmodelslim工具生成的int4量化权重，dtype仍为int8，权重中数值分布在int4范围内。昇腾推理框架例如MindIE-LLM，在将权重加载到npu上前，会对权重进行合并处理，因此实际的内存占用是int4对应的内存大小。


## W8A8, W8A8S量化
量化工具对于每个量化的Linear生成五个参数，参数名称为：weight, input_scale, input_offset, deq_scale, quant_bias  
在safetensors权重文件中，完整的权重名称为Lienar层的名称+参数名称，与W8A16或W4A16类似。

weight为量化后的int8或int4的权重，数据类型为torch.Tensor，dtype为torch.int8，shape和原始浮点的shape一致，记为n, k = weight.shape。
weight_scale为量化的缩放系数，数据类型为torch.Tensor，dtype为torch.float16或torch.float32，在per_channel场景下，shape为[n, 1]，在per_group场景下，shape为[n, k / group_size]。
weight_offset为量化的偏移系数，数据类型为torch.Tensor，dtype和shape和weight_scale一致。对称量化场景下需要构造全0的weight_offset。

反量化的计算公式为:  
per_channel 场景：  
deq_weight = (weight - weight_offset) * weight_scale  
per_group 场景  
weight = weight.reshape((-1, group_size))  
weight_offset = weight_offset.reshape((n * k / group_size, 1))  
weight_scale = weight_scale.reshape((n * k / group_size, 1))  
deq_weight = ((weight - weight_offset) * weight_scale).reshape((n, k))





Ascend Inference Tools，昇腾推理工具链。 【Powered by MindStudio】

**请根据自己的需要进入对应文件夹获取工具，或者点击下面的说明链接选择需要的工具进行使用。**

### 模型推理迁移全流程
![模型推理迁移全流程](/ait_flow.png)

### 大模型推理迁移全流程
![大模型推理迁移全流程](/ait-llm-flow.png)

## 使用说明

1.  [ait](https://gitee.com/ascend/ait/tree/master/ait)

    **一体化推理开发工具**：作为昇腾统一推理工具，提供客户一体化开发工具，支持一站式调试调优，当前包括benchmark、debug、transplt、analyze等组件。

2.  [onnx-modifier](https://gitee.com/ascend/ait/tree/master/onnx-modifier)

    **可视化改图工具**：提供ONNX模型的实时预览、可视化改图功能，从而更方便、快捷、直观地实现ONNX模型的编辑。
3.  [ide](https://gitee.com/ascend/ait/tree/master/ide)

    **一体化推理开发工具IDE插件**：作为一体化推理工具集成IDE插件，当前集成模型转换，Ais_Bench，Compare（一键式精度比对）等组件。

#### 许可证
[Apache License 2.0](LICENSE)

