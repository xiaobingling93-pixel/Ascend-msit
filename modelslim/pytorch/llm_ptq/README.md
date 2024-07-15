
# msmodelslim量化权重格式

msmodelslim llm-ptq工具生成的safetensors量化权重文件包含两个文件，quant_model_weight.safetensors权重文件和quant_model_description.json权重描述文件

msmodelslim工具生成的量化权重均为signed场景，即对应int8数据分布范围为-128~127，对于int4数据分布范围为-8~7。开源权重若为unsigned场景，对于int8可以考虑将weight和offset权重减去128，对于int4则减去8



## W8A16, W4A16量化

量化工具对于每个量化的Linear生成3个参数，参数名称为：`weight`、`weight_scale`、`weight_offset`，在safetensors权重文件中，完整的权重名称为Lienar层的名称+参数名称，例如ChatGLM2-6B量化权重中，`"transformer.encoder.layers.0.self_attention.query_key_value.weight_scale"`，`"transformer.encoder.layers.0.self_attention.query_key_value"`为Linear层的名称，`"weight_scale"`为参数的名称

### 权重说明
`weight`为量化后的int8或int4的权重，数据类型为torch.Tensor，dtype为torch.int8，shape和原始浮点的shape一致，记为n, k = weight.shape，k为hidden_size  
`weight_scale`为量化的缩放系数，数据类型为torch.Tensor，dtype为torch.float32，在per_channel场景下，shape为[n]，在per_group场景下，shape为[n, k / group_size]  
`weight_offset`为量化的偏移系数，数据类型为torch.Tensor，dtype和shape和weight_scale一致。对称量化场景下需要构造全0的weight_offset  

### 反量化计算公式 
per_channel 场景：  
```python
deq_weight = (weight - weight_offset) * weight_scale  
```
per_group 场景： 
```python
weight = weight.reshape((-1, group_size))  
weight_offset = weight_offset.reshape((n * k / group_size, 1))  
weight_scale = weight_scale.reshape((n * k / group_size, 1))  
deq_weight = ((weight - weight_offset) * weight_scale).reshape((n, k))
```

对于W4A16场景，因为原生pytorch不支持torch.int4的tensor类型，开源的模型、量化工具有各自的实现方式，例如ChatGLM2-6B采用两个int4的tensor合并成int8的tensor，0x1,0x2,0x3,0x4合并成0x12,0x34；AutoAWQ采用8个int4的tensor合并成int32的tensor，0x1,0x2...0x8合并成0x86427531，不能整除的部分补0。msmodelslim工具生成的int4量化权重，dtype仍为int8，权重中数值分布在int4范围内。昇腾推理框架例如MindIE-LLM，在将权重加载到npu上前，会对权重进行合并处理，因此实际的内存占用是int4对应的内存大小

代码实现可以参考demo样例MSModelSlimWeightProcessor.weight_process，请根据开源权重的反量化公式和msmdoelslim工具的反量化公式进行相应修改


## W8A8, W8A8S量化
msmodelslim量化工具对于每个量化的Linear生成5个参数，参数名称为：weight, input_scale, input_offset, deq_scale, quant_bias  
在safetensors权重文件中，完整的权重名称为Linear层的名称+参数名称，与W8A16或W4A16类似

### 权重说明
`weight`为量化后的int8或int4的权重，数据类型为torch.Tensor，dtype为torch.int8，shape和原始浮点的shape一致，记为n, k = weight.shape，k为hidden_size   
`input_scale`为激活值量化的缩放系数，数据类型为torch.Tensor，dtype为torch.float16，shape为[1]  
`input_offset`为激活值量化的偏移系数，数据类型为torch.Tensor，dtype和shape和input_scale一致。  
`deq_scale`为反量化缩放系数，数据类型为torch.Tensor，dtype为torch.int64，shape为[n]。注意为了亲和昇腾量化算子，deq_scale的数据在传给量化算子前需要进行特殊处理，可以参考示例代码109行进行处理
`quant_bias`为反量化的偏移系数，数据类型为torch.Tensor，dtype为torch.int32，shape为[n]  

### 量化、反量化计算公式 
```python
input_int8 = input_fp / input_scale + input_offset  
output_int8 = input_int8 * weight + quant_bias  
output_deq = output_int8 * deq_scale  
```

代码实现可以参考demo样例MSModelSlimWeightProcessor.weight_activation_process，请根据开源权重的计算公式和msmdoelslim工具的计算公式进行相应修改

## smooth quant
msmodelslim量化工具使用smooth quant后，对于每个norm层，生成2个参数，module.weight和module.bias。完整的权重名称为norm层的名称+参数名称，例如ChatGLM2-6B量化权重中，`"transformer.encoder.layers.0.input_layernorm.module.weight"`，`"transformer.encoder.layers.0.input_layernorm"`为norm层的名称，`"module.weight"`为量化参数名称

msmodelslim量化工具集成的smooth quant算法针对norm层后的Linear层进行smooth平滑操作，而不是所有Linear层。采取这种量化方案的优势在于可以将原本乘在激活值上的scale等价转移到原始浮点模型norm层的权重`norm.weight`上，从而避免额外引入算子带来的性能开销

`module.weight`为scale后的norm.weight，数据类型、dtype、shape和norm.weight一致  
`module.bias`为引入module.weight后带来的偏移系数，数据类型、dtype、shape和norm.weight一致  
 
为了适配几种特殊的回退情况，msmodelslim生成的smooth quant权重中还包含原始浮点权重norm层的权重，norm.weight。如果开源量化权重不涉及回退场景，设置为None即可

代码实现可以参考demo样例MSModelSlimWeightProcessor.anti_outlier_process，请根据开源权重的设计方案和msmdoelslim工具的设计方案进行相应修改


## KV Cache量化
msmodelslim工具提供的KV Cache量化采用int8量化。对于每个attention层，生成4个参数，`k_proj.kv_cache_scale`, `k_proj.kv_offset`, `v_proj.kv_cache_scale`, `v_proj.kv_cache_offset`。对于qkv合并或kv合并的场景，完整的四个参数的名称为合并的Linear名称+参数名；对于qkv分离场景，k_proj的scale、offset完整的参数名称为k对应Linear名称+参数名称，v_proj的scale、offset完整的参数名称为v对应Linear名称+参数名称。例如`"transformer.encoder.layers.0.query_key_value.k_proj.kv_cache_scale"`，`"transformer.encoder.layers.0.query_key_value"`为qkv合并的Linear层名称，`"k_proj.kv_cache_scale"`为参数名称

### 权重说明
`kv_cache_scale`为kvcache量化scale的缩放系数，数据类型为torch.Tensor，dtype为torch.float32或torch.float16，shape为kv channel的size，如果是qkv分开场景，则为k或v层linear的n维（见上文**w8a16，w4a16量化** 章节 **权重说明** weight的shape说明）
`kv_scale_offset`为kvcache量化scale的偏移系数，数据类型、dtype、shape和scale一致


计算公式：  
量化 
```python
cache_int = cache_fp / cache_scale + cache_offset
```
反量化 
```python
cache_deq = (cache_int - cache_offset) * cache_scale
```

代码实现可以参考demo样例MSModelSlimWeightProcessor.kv_cache_process，请根据开源权重的计算公式和msmdoelslim工具的计算公式进行相应修改

