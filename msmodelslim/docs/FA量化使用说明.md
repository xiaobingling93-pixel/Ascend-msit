## fa3量化 

**Flash Attention 3（FA）**，在KV-Cache的基础上增强了在硬件设备上的利用率，提升了整体在推理场景中的计算效率，以低精度的数据格式完成更快的处理和更少的内存占用。

### 前提条件

前提条件参考[[大模型量化的前提条件](https://gitee.com/ascend/msit/blob/master/msmodelslim/msmodelslim/pytorch/llm_ptq/README.md#%E5%89%8D%E6%8F%90%E6%9D%A1%E4%BB%B6)]

说明：仅Atlas 800I A2和Atlas 800I A3推理产品支持fa3量化功能。

### 功能实现流程

关键步骤说明如下：

#### 1.确定Attention:
需确定模型基于哪一个Attention进行实现，以Qwen2.5模型为例，有Qwen2Attention、Qwen2FlashAttention2和Qwen2SdpaAttention三种Attention。如未特殊指定，默认为Qwen2Attention。

#### 2.修改modeling文件：

（1）找到对应版本的modeling文件：

每个模型的modeling文件路径和对应版本都可以在权重路径下的config里查到。以Qwen2.5_70B为例，权重目录下的config如下所示，config中`model_type="qwen2"`，`transformers_version="4.43.1"`。那么就可以去transformer库里找4.43.1版本的[modeling_qwen2.py](https://github.com/huggingface/transformers/blob/v4.43.1/src/transformers/models/qwen2/modeling_qwen2.py)

```python
{
  "architectures": [
    "Qwen2ForCausalLM"
  ],
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "hidden_act": "silu",
  "hidden_size": 8192,
  "initializer_range": 0.02,
  "intermediate_size": 29568,
  "max_position_embeddings": 32768,
  "max_window_layers": 70,
  "model_type": "qwen2",
  "num_attention_heads": 64,
  "num_hidden_layers": 80,
  "num_key_value_heads": 8,
  "rms_norm_eps": 1e-06,
  "rope_theta": 1000000.0,
  "sliding_window": 131072,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.43.1",
  "use_cache": true,
  "use_sliding_window": false,
  "vocab_size": 150000
}
```

（2）修改modeling文件：

- 添加引用依赖：

```python
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.fa_quant import FAQuantizer 
from msmodelslim import logger 
```

- 在使用的Attention处调用工具：

在init初始化处添加：

```python
self.fa_quantizer = FAQuantizer(self.config, logger)
```

在forward部分添加:

```python
query_states = self.fa_quantizer.quant(query_states, qkv="q")
key_states = self.fa_quantizer.quant(key_states, qkv="k")
value_states = self.fa_quantizer.quant(value_states, qkv="v")
```
注意：新增的query_states、key_states和value_states的量化代码需放置在`if past_key_value is not None:`代码块之后，`key_states = repeat_kv(key_states, self.num_key_value_groups)`代码块之前。如果某些attention结构（如mha）没有`key_states = repeat_kv(key_states, self.num_key_value_groups)`代码块，则将量化代码放置在`if past_key_value is not None:`代码块之后即可。

- 整体修改如下：

```python
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.fa_quant import FAQuantizer 
from msmodelslim import logger 


class Qwen2Attention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: Qwen2Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        ...
        # 其他未修改的代码部分
        ...
        
    	# 新增的代码部分
        # --------------------------------------------------
    	self.fa_quantizer = FAQuantizer(self.config, logger)
        # --------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        ...
        # 其他未修改的代码部分
        ...
        
        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            
		    # 新增的代码部分
        # --------------------------------------------------
        query_states = self.fa_quantizer.quant(query_states, qkv="q")
        key_states = self.fa_quantizer.quant(key_states, qkv="k")
        value_states = self.fa_quantizer.quant(value_states, qkv="v")
        # --------------------------------------------------
       
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
       
        ...
        # 其他未修改的代码部分
        ...

```

**注意**：部分模型在transformers的库中对其组件的依赖是采用的相对路径，在改写了modeling文件之后需要将这部分相对路径的导入依赖改成绝对路径，例如：
```python
"""
# 修改前的导入方式
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache, StaticCache
from ...modeling_attn_mask_utils import AttentionMaskConverter
from ...modeling_flash_attention_utils import _flash_attention_forward
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
"""
# 修改后的导入方式
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
```

（3）修改完毕后的modeling文件需放在模型权重路径下，对config文件进行修改来指定模型加载时所使用的modeling文件。假设修改后的modeling文件为`modeling_qwen2_fa3.py`，config文件做如下修改：

```json
{
  "architectures": [
    "Qwen2ForCausalLM"
  ],
    // 新增配置
    // --------------------------------------------------
    "auto_map": {
    "AutoModelForCausalLM": "modeling_qwen2_fa3.Qwen2ForCausalLM"
    // --------------------------------------------------
    ...
    // 其他未修改的代码部分
    ...

```
**注意**：在量化脚本里面通过transformers库对模型进行加载时，调用`from_pretrained`函数时一定要指定`trust_remote_code=True`让修改后的modeling文件能够正确的被加载。

#### 3.配置config:

`config = QuantConfig().fa_quant()`

在QuantConfig初始化中完成核心参数`(w_bit，a_bit，disable_names，disable_last_linear，dev_type，dev_id)`的配置后，如果需要使用FA量化的新特性，通过调用QuantConfig的`fa_quant` 函数完成配置。

具体的参数说明如下：

| **量化类型**                          | **需要配置的参数列表**                                       | **调用示例**                                                 |
| ------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| fa_quant(fa_amp=5) | fa_amp用于配置自动精度回退，根据想要回退的layer的数量来设置。<br>数据类型为int，默认值为0。数据取值范围是大于等于0，并且小于等于模型layer数量，如果超出模型的layer数量将会取模型的最大layer数量为回退层数。 | quant_config=QuantConfig(w_bit=8,  a_bit=8, disable_names=disable_names,dev_type='npu',dev_id=0).fa_quant(fa_amp=5)|

### 量化步骤（以Qwen2.5-7B为例）

1. 用户自行准备模型、权重文件和校准数据，将修改好的modeling文件和config放入权重目录下，本样例以Qwen2.5-7B为例，目录示例如下：

```bash

├── config.json

├── modeling_qwen2.py

├── generation_config.json

├── merges.txt

├── model-00001-of-00004.safetensors

├── model-00002-of-00004.safetensors

├── model-00003-of-00004.safetensors

├── model-00004-of-00004.safetensors

├── model.safetensors.index.json

├── README.md

├── tokenizer_config.json.py

├── tokenizer.json

├── vocab.json

```

2. 新建模型的量化脚本quant.py，编辑quant.py文件，根据实际的量化场景导入样例代码，参考加粗字体信息提示，并根据实际情况进行修改。

注：fa3量化目前仅支持W8A8 per_channel量化场景和lowbit算法，W8A8 per_channel量化场景导入的样例见下文FA3精度调优处，lowbit算法的代码样例请参考w8a8量化场景。


3. 启动模型量化任务，并在指定的输出目录获取模型量化参数，量化后权重文件的介绍请参见量化后权重文件，若使用MindIE进行后续的推理部署任务，请保存为safetensors格式，具体请参见MindIE的“加速库支持模型列表”章节中已适配量化的模型。

```python
python3 quant.py
```

###  量化后权重文件

- **npy格式**

当save_type设置为['numpy']或不设置时，量化权重会保存为npy文件，npy储存格式为字典，其中key值为各层Linear的名字，例如Qwen2.5-7B模型的transformer.encoder.layers.0.self_attention.query_key_value，value值为第0层query_key_value的Linear权重。

```bash

├── anti_fp_norm.npy  #Qwen模型已启用离群抑制功能，具体操作请参见使用离群值抑制功能，将会生成此文件。antioutlier算法生成浮点权重中的norm层权重文件，用于量化层的input和post norm的权重适配

├── deq_scale.npy    #W8A8量化的量化参数权重文件，Tensor数据类型为int64，deq_scale已针对量化算子进行数据类型转换，可直接适配算子。在量化BF16模型情况下，数据类型不会转换为int64，仍然为float32

├── fa_quant_offset.npy    #fa3量化的激活值量化偏移值参数文件，Tensor数据类型为bfoat16或float16

├── fa_quant_scale.npy   #fa3量化的激活值量化缩放因子参数文件，Tensor数据类型为bfoat16或float16

├── input_offset.npy  #W8A8量化的激活值量化偏移值权重文件，Tensor数据类型为float32

├── input_scale.npy   #W8A8量化的激活值量化缩放因子权重文件，Tensor数据类型为float32

├── quant_bias.npy   #W8A8量化的量化参数权重文件，Tensor数据类型为int32，quant_bias已考虑原始浮点模型linear层的bias值

├── quant_weight.npy  #量化权重文件，Tensor数据类型为int8

```

推理部署时读取上述文件的示例代码：quant_param_dict = np.load("xxx.npy", allow_pickle=True).item()。



- **safetensors格式**

当save_type设置为['safe_tensor']时，量化权重会保存为safetensors文件和json描述文件。
  
- safetensors中储存格式为字典，包含量化权重和量化不修改的浮点权重。其中量化权重的key值为各层Linear的名字加上对应权重的名字，module.weight和module.bias对应anti_fp_norm.npy，weight对应quant_weight.npy，quant_bias对应quant_bias.npy等以此类推。例如Qwen2.5-7B模型的model.layers.0.self_attn.q_proj.deq_scale对应npy格式权重中deq_scale.npy中的model.layers.0.self_attn.q_proj;

```python
# qwen模型量化生成的权重文件部分内容
{
  "model.embed_tokens.weight": tensor([...]),
  "model.layers.0.self_attn.q_proj.weight": tensor([...]),
  "model.layers.0.self_attn.q_proj.input_scale": tensor([...]),
  "model.layers.0.self_attn.q_proj.input_offset": tensor([...]),
  "model.layers.0.self_attn.q_proj.quant_bias": tensor([...]),
  "model.layers.0.self_attn.q_proj.deq_scale": tensor([...]),
  "model.layers.0.self_attn.k_proj.weight": tensor([...]),
   ...
   "model.layers.0.self_attn.fa_q.scale": tensor([...]),
   "model.layers.0.self_attn.fa_q.offset": tensor([...]),
   "model.layers.0.self_attn.fa_k.scale": tensor([...]),
   "model.layers.0.self_attn.fa_k.offset": tensor([...]),
   "model.layers.0.self_attn.fa_v.scale": tensor([...]),
   "model.layers.0.self_attn.fa_v.offset": tensor([...]),
   ...
}
```


- json描述文件中储存的量化权重的总体类型model_quant_type，是否启用fa3量化fa_quant_type，和其中各个权重的类型，来自原始浮点权重则为FLOAT，来自W8A8量化则为W8A8。

```python
{
  "model_quant_type": "W8A8",                                # 整体量化类型为w8a8量化
  "fa_quant_type": "FAQuant",								                 # 量化过程开启了fa3量化
  "model.embed_tokens.weight": "FLOAT",                      # 来自原始浮点模型的embed_tokens权重
  "model.layers.0.self_attn.q_proj.weight": "W8A8",          # 量化新增的第0层self_attn.q_proj的quant_weight
  "model.layers.0.self_attn.q_proj.input_scale": "W8A8",     # 量化新增的第0层self_attn.q_proj的input_scale
  "model.layers.0.self_attn.q_proj.input_offset": "W8A8",    # 量化新增的第0层self_attn.q_proj的input_offset
  "model.layers.0.self_attn.q_proj.quant_bias": "W8A8",      # 量化新增的第0层self_attn.q_proj的quant_bias
  "model.layers.0.self_attn.q_proj.deq_scale": "W8A8",       # 量化新增的第0层self_attn.q_proj的deq_scale
  "model.layers.0.self_attn.k_proj.weight": "W8A8",          # 量化新增的第0层self_attn.k_proj的quant_weight
   ...
   "model.layers.0.self_attn.fa_q.scale": "FAQuant",         # 量化新增的第0层self_attn的query_states的scale
   "model.layers.0.self_attn.fa_q.offset": "FAQuant",        # 量化新增的第0层self_attn的query_states的offset
   "model.layers.0.self_attn.fa_k.scale": "FAQuant",         # 量化新增的第0层self_attn的key_states的scale
   "model.layers.0.self_attn.fa_k.offset": "FAQuant",        # 量化新增的第0层self_attn的key_states的offset
   "model.layers.0.self_attn.fa_v.scale": "FAQuant",         # 量化新增的第0层self_attn的key_states的scale
   "model.layers.0.self_attn.fa_v.offset": "FAQuant",        # 量化新增的第0层self_attn的key_states的offset
   ...
}
```

###  FA3精度调优

#### 量化脚本（NPU）
```python
"""
1、导入相关依赖
"""
import json
import torch
import torch_npu # 如果需要使用npu进行量化
from transformers import AutoTokenizer, AutoModelForCausalLM
from modelslim.pytorch.llm_ptq.anti_outlier import AntiOutlierConfig, AntiOutlier
from modelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig
from precision_tool.precision_tool import PrecisionTest # precision_tool用于伪量化测精度

SEQ_LEN_OUT = 100
batch_size = 1

# 如果使用npu进行量化需开启二进制编译，避免在线编译算子
torch.npu.set_compile_mode(jit_compile=False)
option = {}
option["NPU_FUZZY_COMPILE_BLACKLIST"] = "ReduceProd"
torch.npu.set_option(option)

"""
2、导入相关模型
"""
model_path = '/data/model_path' # 原始浮点模型路径

# 这里以8卡32G机器为例，6卡量化。
model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_path,
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True,
        device_map="auto",
        max_memory={0:"28GiB",1:"28GiB",2:"28GiB",3:"28GiB",4:"28GiB",5:"28GiB",6:"0GiB",7:"0GiB"}
    ).eval()

tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=model_path, 
    trust_remote_code=True,
    device_map="auto",
)
tokenizer.pad_token = tokenizer.eso_token

"""
数据集测原始模型浮点精度（此示例中选择的是boolq）
"""
precision_test = PrecisionTest(model, tokenizer, "boolq", batch_size, "npu")
precision_test.test()

"""
3、获取校准数据
"""
def get_calib_dataset(tokenizer, calib_list, device=model.device):
    calib_dataset = []
    for calib_data in calib_list:
        inputs = tokenizer(calib_data, return_tensors='pt')
        calib_dataset.append([
            inputs.data['input_ids'].to(device),
            inputs.data['attention_mask'].to(device)
        ])
             
    return calib_dataset

entry = "/path/to/calib_dataset" # 此示例中校准数据选取50条左右boolq数据
with open(entry, 'r') as file:
    calib_set = json.load(file)
dataset_calib = get_calib_dataset(tokenizer, calib_set)
"""
4、离群值抑制AntiOutlier(w8a8)
"""
anti_config = AntiOutlierConfig(anti_method="m3", dev_type="npu", dev_id=model.device.index)
anti_outlier = AntiOutlier(model, calib_data=dataset_calib, cfg=anti_config)
anti_outlier.process()

"""
5、回退层设置
"""
"""
因为一些量化后的网络层对精度影响太大了，所以需要让这些网络层使用浮点权重进行计算， disable_names中为需要进行回退的网络层。
"""
disable_names = []
num_layers = 80
disable_idx_lst = list(range(num_layers))
for layer_index in disable_idx_lst:
    down_proj_name = "model.layers.{}.mlp.down_proj".format(layer_index)
    disable_names.append(down_proj_name)
"""
6、执行PTQ量化校准 + 存储量化参数用于部署
"""
quant_config = QuantConfig(
    a_bit=8,
    w_bit=8,
    disable_names=disable_names,
    dev_type='npu',
    dev_id=model.device.index,
    act_method=3,
    pr=1.0,
    w_sym=True,
    mm_tensor=False
).fa_quant(fa_amp=0) #调用fa_quant之后默认开启FA量化，fa_amp可设置自动回退层数

calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L5')  # disable_level: 自动回退n个linear
calibrator.run()  # 执行PTQ量化校准
calibrator.save('/save/path', save_type=["safe_tensor", "numpy"]) # "safe_tensor"对应safetensors格式权重，"numpy"对应npy格式权重

"""
数据集测伪量化模型精度（此示例中选择的是boolq）
"""
precision_test = PrecisionTest(model, tokenizer, "boolq", batch_size, "npu")
precision_test.test()
```

#### 本文仅给出FA3场景下Llama3.1-70B和Qwen2.5-72B的量化推荐配置，可按实际情况进行参数调整，详见[精度调优策略](https://gitee.com/ascend/msit/tree/dev/msmodelslim/docs/w8a8精度调优策略.md) 。

#### Llama3.1-70B 量化参数设置

- 离群值抑制(AntiOutlier) ：anti_method = "m3"
```python
anti_config = AntiOutlierConfig(anti_method="m3", dev_type="npu", dev_id=model.device.index)
```

- 量化参数(QuantConfig)

激活值量化方法：act_method = 3
```python
quant_config = QuantConfig(
    a_bit=8,
    w_bit=8,
    disable_names=disable_names,
    dev_type='npu',
    dev_id=device_id,
    act_method=3,
    pr=1.0,
    w_sym=True,
    mm_tensor=False
).fa_quant(fa_amp=0)

calibrator = Calibrator(
    model, 
    quant_config, 
    calib_data=dataset_calib, 
    disable_level='L5'
)  
```

- 校准数据(calib_set)
使用50条左右boolq数据集校准 

- 量化回退(disable_names)
（1）disable_level='L5': 自动回退5个layer
<br>

（2）回退所有down层：
```python
disable_names = []
num_layers = 80
disable_idx_lst = list(range(num_layers))
for layer_index in disable_idx_lst:
    down_proj_name = "model.layers.{}.mlp.down_proj".format(layer_index)
    disable_names.append(down_proj_name)
```
（3）（可选）调用fa_quant时设置回退层数。本模型无需设置该参数精度即可达标。
```python
fa_quant(fa_amp=5)
```

#### Qwen2.5-72B 量化参数设置

- （可选）离群值抑制(AntiOutlier) ：无需离群值抑制精度即可达标

- 量化参数(QuantConfig)

激活值量化方法：act_method = 1
```python
quant_config = QuantConfig(
    a_bit=8,
    w_bit=8,
    disable_names=disable_names,
    dev_type='npu',
    dev_id=device_id,
    act_method=1,
    pr=1.0,
    w_sym=True,
    mm_tensor=False
).fa_quant(fa_amp=0)

calibrator = Calibrator(
    model, 
    quant_config, 
    calib_data=dataset_calib, 
    disable_level='L0'
)  
```

- 校准数据(calib_set)
使用50条左右boolq数据集校准 

- 量化回退(disable_names)
（1）（可选）disable_level='L0': 本模型设置L0精度即可达标。
<br>（2）回退所有down层：
```python
disable_names = []
num_layers = 80
disable_idx_lst = list(range(num_layers))
for layer_index in disable_idx_lst:
    down_proj_name = "model.layers.{}.mlp.down_proj".format(layer_index)
    disable_names.append(down_proj_name)
```
（3）（可选）调用fa_quant时设置回退层数。本模型无需设置该参数精度即可达标。
```python
fa_quant(fa_amp=5)
```
