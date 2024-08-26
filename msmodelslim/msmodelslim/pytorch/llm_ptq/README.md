## 大模型量化

大模型量化工具将高位浮点数转为低位的定点数，例如16bit降低到8bit，直接减少模型权重的体积，生成量化参数和权重文件。在无需训练成本的前提下，完成大模型的训练后压缩并最大程度保障其精度。

### 前提条件

- 仅支持在以下产品中使用。
    - Atlas 推理系列产品（Ascend 310P处理器）。
    - Atlas 训练系列产品。
    - Atlas A2训练系列产品/Atlas 800I A2推理产品。

- 已参考环境准备，完成CANN开发环境的部署、以及PyTorch 2.1.0及以上版本的框架和npu插件、Python环境变量配置。
- 大模型量化工具须执行命令安装如下依赖。
  如下命令如果使用非root用户安装，需要在安装命令后加上--user，例如：pip3 install onnx --user。
```
pip3 install numpy==1.25.2
pip3 install transformers        #需大于等于4.29.1版本，LLaMA模型需指定安装4.29.1版本
pip3 install accelerate==0.21.0  #若需要使用NPU多卡并行方式对模型进行量化，需大于等于0.28.0版本
pip3 install tqdm==4.66.1
```
- （可选）如果需要在大模型量化工具中使用NPU多卡并行的方式对模型进行量化，需关闭NPU设备中的虚拟内存，并手动配置量化将会执行的设备序列环境。
```
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False # 关闭NPU的虚拟内存
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 #配置量化将会执行的设备序列环境
```
说明
仅Atlas 训练系列产品和Atlas A2训练系列产品/Atlas 800I A2推理产品支持此功能。

### 功能实现流程

关键步骤说明如下：

1. 用户准备原始模型和校准数据。

2. 可选：使用离群值抑制功能对LLM模型进行离群值抑制，可参考精度保持策略选择是否启用。
    - 使用AntiOutlierConfig生成离群值抑制配置。
    - 调用AntiOutlier接口，将模型、校准数据等传入，生成抑制器。
    - 调用抑制器的process()方法对原始模型进行离群值抑制。

3. 使用QuantConfig生成量化配置。

4. 根据原始模型、量化配置和校准数据，调用Calibrator接口构建量化校准对象。

5. 调用生成的量化校准对象的run()方法对原始模型进行量化。

6. 调用生成的量化校准对象的save()接口保存量化后的模型，包括模型量化权重和模型相关参数，用于后续量化模型的部署任务，，具体请参见MindIE的“加速库支持模型列表”章节中已适配量化的模型。

### 量化步骤（以ChatGLM2-6B为例）

1. 用户自行准备模型、权重文件和校准数据，本样例以ChatGLM2-6B为例，目录示例如下：
```
├── config.json
├── configuration chatglm.py
├── modeling_chatglm.py
├── pytorch_model-00001-of-00007.bin
├── pytorch_model-00002-of-00007.bin
├── pytorch_model-00003-of-00007.bin
├── pytorch_model-00004-of-00007.bin
├── pytorch_model-00005-of-00007.bin
├── pytorch_model-00006-of-00007.bin
├── pytorch_model-00007-of-00007.bin
├── pytorch_model.bin.index.json
├── quantization.py
├── README.md
├── tokenization_chatglm.py
├── tokenizer.model
├── tokenizer_config.json
```

2. ChatGLM2-6B模型进行量化前请执行如下命令安装所需依赖，若运行量化工具过程中提示缺失某个依赖，请根据提示安装。
```
pip3 install protobuf==4.24.1
pip3 install sentencepiece==0.1.99
pip3 install sympy==1.11.1
```

3. 新建模型的量化脚本quant.py，编辑quant.py文件，根据实际的量化场景导入样例代码，参考加粗字体信息提示，并根据实际情况进行修改。
    - W8A8 per_channel量化场景导入的样例代码如下，kvcache、lowbit算法以及per_token算法量化场景导入的代码样例请参考w8a8量化场景。
```
# 导入相关依赖
import torch 
import torch_npu   # 若需要cpu上进行量化，可忽略此步骤
from transformers import AutoTokenizer, AutoModel

# for local path
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path='./chatglm2', trust_remote_code=True)
model = AutoModel.from_pretrained(
    pretrained_model_name_or_path='./chatglm2',
    trust_remote_code=True,
  ).npu()    # 若在npu上进行多卡量化时，需要先参考前提条件进行配置，并配置device_map='auto',创建model时需去掉.npu()；若在cpu上进行量化时，需要配置torch_dtype=torch.float32，创建model时需去掉.npu()
# 准备校准数据，请根据实际情况修改
calib_list = ["中国的首都在哪里？",
              "请做一首诗歌：",
              "我想要学习python，该怎么学习？",
              "请帮我写一篇关于大模型推理优化的任职报告：",
              "中国最值得去的几个景点"]
#获取校准数据函数定义
def get_calib_dataset(tokenizer, calib_list):
    calib_dataset = []
    for calib_data in calib_list:
        inputs = tokenizer([calib_data], return_tensors='pt').to(model.device)   
        print(inputs)
        calib_dataset.append([inputs.data['input_ids'], inputs.data['attention_mask']])     
        return calib_dataset

dataset_calib = get_calib_dataset(tokenizer, calib_list)  #校准数据获取

# 量化配置，请根据实际情况修改
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig    # 导入量化配置接口
# 使用QuantConfig接口，配置量化参数，并返回量化配置实例
quant_config = QuantConfig(
    a_bit=8, 
    w_bit=8,       
    disable_names=['transformer.encoder.layers.0.self_attention.query_key_value','transformer.encoder.layers.0.self_attention.dense', 'transformer.encoder.layers.0.mlp.dense_h_to_4h'], 
    dev_id=model.device.index， 
    dev_type='npu',   # 在cpu进行量化时，需配置参数dev_type='cpu'，并取消dev_id=model.device.index参数的配置
    act_method=3,
    pr=0.5, 
    mm_tensor=False
  )  
#使用Calibrator接口，输入加载的原模型、量化配置和校准数据，定义校准
calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')  
calibrator.run()     #使用run()执行量化
calibrator.save('./quant_weight', save_type=[ 'numpy', 'safe_tensor'])      #使用save()保存模型量化参数，请根据实际情况修改路径
print('Save quant weight success!')
```

    - W8A16或W4A16 per_channel量化场景导入的样例代码如下，MinMax算法、HQQ算法、GPTQ算法、AWQ算法以及w4a16 per-group量化场景导入的代码样例请参考w8a16或w4a16量化场景。
```
# 导入相关依赖
import torch
import torch_npu   # 若需要cpu上进行量化，可忽略此步骤
from transformers import AutoTokenizer, AutoModel

# for local path
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path='./chatglm2', trust_remote_code=True) 
model = AutoModel.from_pretrained(
    pretrained_model_name_or_path='./chatglm2',
    trust_remote_code=True,
    ).npu()    # 若在npu上进行多卡量化时，需要先参考前提条件进行配置，并配置device_map='auto'，创建model时需去掉.npu()；若在cpu上进行量化时，需要配置torch_dtype=torch.float32，创建model时需去掉.npu()
# 准备校准数据，请根据实际情况修改，W8A16 Label-Free模式下请忽略此步骤
calib_list = ["中国的首都在哪里？",
              "请做一首诗歌：",
              "我想要学习python，该怎么学习？",
              "请帮我写一篇关于大模型推理优化的任职报告：",
              "中国最值得去的几个景点"]
#获取校准数据函数定义
def get_calib_dataset(tokenizer, calib_list):
    calib_dataset = []
    for calib_data in calib_list:
        inputs = tokenizer([calib_data], return_tensors='pt').to(model.device)
        print(inputs)
        calib_dataset.append([inputs.data['input_ids'], inputs.data['attention_mask']])
    return calib_dataset

dataset_calib = get_calib_dataset(tokenizer, calib_list)  #校准数据获取

# 量化配置，请根据实际情况修改
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig    # 导入量化配置接口
# 使用QuantConfig接口，配置量化参数，并返回量化配置实例
quant_config = QuantConfig(
    w_bit=8,     # W4A16场景下，w_bit值需配置为4。在W4A16的per_group场景下，需参考W4A16的per_group量化场景参数进行设置。
    a_bit=16,         
    disable_names=[], 
    dev_id=model.device.index， 
    dev_type='npu',   # 在cpu进行量化时，需配置参数dev_type='cpu'，并取消参数dev_id=model.device.index的配置
    w_sym=False, 
    mm_tensor=False
  )  
#使用Calibrator接口，输入加载的原模型、量化配置和校准数据，定义校准
calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')  calibrator.run()     #使用run()执行量化
calibrator.save('./quant_weight', save_type=[ 'numpy', 'safe_tensor'])      #使用save()保存模型量化参数，请根据实际情况修改路径
print('Save quant weight success!')
```

4. 启动模型量化任务，并在指定的输出目录获取模型量化参数，量化后权重文件的介绍请参见量化后权重文件，若使用MindIE进行后续的推理部署任务，请保存为safetensors格式，具体请参见MindIE的“加速库支持模型列表”章节中已适配量化的模型。
```
python3 quant.py
```
量化任务完成后，可能会存在模型精度下降的情况，可以参考精度保持策略进行配置优化减少精度损耗。

### 量化后权重文件
- npy格式
当save_type设置为['numpy']或不设置时，量化权重会保存为npy文件，npy储存格式为字典，其中key值为各层Linear的名字，例如ChatGLM2-6B模型的transformer.encoder.layers.0.self_attention.query_key_value，value值为第0层query_key_value的Linear权重。
```
├── anti_fp_norm.npy   #LLaMA模型且已启用离群抑制功能，具体操作请参见使用离群值抑制功能，将会生成此文件。antioutlier算法生成浮点权重中的norm层权重文件，用于量化层的input和post norm的权重适配
├── deq_scale.npy      #W8A8量化和稀疏量化的量化参数权重文件，Tensor数据类型为int64，deq_scale已针对量化算子进行数据类型转换，可直接适配算子。在量化BF16模型情况下，数据类型不会转换为int64，仍然为float32
├── input_offset.npy   #W8A8量化和稀疏量化的激活值量化偏移值权重文件，Tensor数据类型为float32
├── input_scale.npy    #W8A8量化和稀疏量化的激活值量化缩放因子权重文件，Tensor数据类型为float32
├── kv_cache_offset.npy    #kv cache量化参数文件，kv linear激活值量化偏移值权重文件，Tensor数据类型为float32
├── kv_cache_scale.npy   #kv cache量化参数文件，kv linear激活值量化缩放因子权重文件，Tensor数据类型为float32
├── quant_bias.npy     #W8A8量化和稀疏量化的量化参数权重文件，Tensor数据类型为int32，quant_bias已考虑原始浮点模型linear层的bias值
├── quant_weight.npy   #量化权重文件，Tensor数据类型为int8
├── weight_offset.npy  #w8a16和w4a16权重量化参数文件，Tensor数据类型为float32
├── weight_scale.npy   #w8a16和w4a16权重量化参数文件，Tensor数据类型为float32
```
推理部署时读取上述文件的示例代码：quant_param_dict = np.load("xxx.npy", allow_pickle=True).item()。

- safetensors格式
当save_type设置为['safe_tensor']时，量化权重会保存为safetensors文件和json描述文件，

    - safetensors中储存格式为字典，包含量化权重和量化不修改的浮点权重。其中量化权重的key值为各层Linear的名字加上对应权重的名字，module.weight和module.bias对应anti_fp_norm.npy，weight对应quant_weight.npy，quant_bias对应quant_bias.npy等以此类推。例如ChatGLM2-6B模型的transformer.encoder.layers.0.self_attention.query_key_value.deq_scale对应npy格式权重中deq_scale.npy中的transformer.encoder.layers.0.self_attention.query_key_value。
```
# llama模型稀疏量化生成的权重文件部分内容
{
  "model.embed_tokens.weight": tensor([...]),
  "model.layers.0.self_attn.q_proj.weight": tensor([...]),
  "model.layers.0.self_attn.q_proj.input_scale": tensor([...]),
  "model.layers.0.self_attn.q_proj.input_offset": tensor([...]),
  "model.layers.0.self_attn.q_proj.quant_bias": tensor([...]),
  "model.layers.0.self_attn.q_proj.deq_scale": tensor([...]),
  "model.layers.0.self_attn.k_proj.weight": tensor([...]),
 ...
}
```
    - json描述文件中储存的量化权重的总体类型model_quant_type，是否启用kvcache量化kv_cache_type，和其中各个权重的类型，来自原始浮点权重则为FLOAT，来自W8A8量化则为W8A8，来自稀疏量化则为W8A8S，来自压缩则为W8A8SC。
```
# llama模型稀疏量化生成的json描述文件部分内容
{
  "model_quant_type": "W8A8S",                               # 整体量化类型为稀疏量化
  "model.embed_tokens.weight": "FLOAT",                      # 来自原始浮点模型的embed_tokens权重
  "model.layers.0.self_attn.q_proj.weight": "W8A8S",         # 量化新增的第0层self_attn.q_proj的quant_weight
  "model.layers.0.self_attn.q_proj.input_scale": "W8A8S",    # 量化新增的第0层self_attn.q_proj的input_scale
  "model.layers.0.self_attn.q_proj.input_offset": "W8A8S",   # 量化新增的第0层self_attn.q_proj的input_offset
  "model.layers.0.self_attn.q_proj.quant_bias": "W8A8S",     # 量化新增的第0层self_attn.q_proj的quant_bias
  "model.layers.0.self_attn.q_proj.deq_scale": "W8A8S",      # 量化新增的第0层self_attn.q_proj的deq_scale
  "model.layers.0.self_attn.k_proj.weight": "W8A8S",         # 量化新增的第0层self_attn.k_proj的quant_weight
 ...
}
```

### 精度保持策略

在量化权重生成后，可以使用伪量化模型进行推理，检验伪量化精度是否正常。伪量化是指通过torch，通过浮点运算完成量化模型运算逻辑，运算过程中的数据和真实量化的数据差异只在算子精度上。如果伪量化精度不满足预期，真实量化结果也将无法满足预期。在调用Calibrator.run()方法后，构建Calibrator时传入的model会被替换为伪量化模型，可以直接调用进行前向推理，用来测试对话效果。如果伪量化结果不理想，可以参考以下手段进行调优：

1. 调整校准数据集：量化模型权重生成对校准数据集有一定依赖，需要根据模型运行场景选取适当的校准数据集。在伪量化精度较差时，可以适当增加校准数据集的数量。

2. 设置Calibrator接口中的“disable_level”参数：配置Calibrator接口中的自动回退等级，可以设置为L0、L1，L2等，依次回退的线性层个数为0、1、2等，在模型精度损失较大时可以适当提升回退等级。
以ChatGLM2-6B为例：

观察到模型伪量化对话效果不理想，考虑进行回退操作。将disable_level设置为L1，生成量化权重。导出的量化权重缺少了key值'transformer.encoder.layers.0.mlp.dense_4h_to_h'对应的权重数据，则该线性层被回退。

如果需要回退整层layer，需要进一步生成量化权重。缺少的linear位于第0层，在QuantConfig接口中的“disable_names”增加该层其余的线性层：'transformer.encoder.layers.0.self_attention.query_key_value','transformer.encoder.layers.0.self_attention.dense', 'transformer.encoder.layers.0.mlp.dense_h_to_4h'。再次生成的量化权重即为整层layer回退的量化权重。

3. 引入离群值抑制AntiOutlier：在模型加载和模型量化之间插入离群值抑制代码，对模型进行离群值抑制，并调用PyTorch接口model.save_pretrained，保存离群值抑制后的浮点模型。
以Llama13B为例：
```
# 离群值抑制
print("outlier suppression start...")
anti_config = AntiOutlierConfig(
    anti_method="m2",
    dev_type='cpu'   # 在npu进行量化时，则需要配置以下参数dev_type='npu'，dev_id=model.device.index。其中dev_id为正确设备号
)
anti_outlier = AntiOutlier(model, calib_data=dataset_calib, cfg=anti_config)
anti_outlier.process()
print("outlier suppression success...")
# save float weight
model.save_pretrained("./llama2-13b_outlier")
```

4. 配置QuantConfig接口中的“pr”：当pr设置为0.5时，导出的量化权重在一定范围内存在随机性，设置为1.0时可以避免随机性。