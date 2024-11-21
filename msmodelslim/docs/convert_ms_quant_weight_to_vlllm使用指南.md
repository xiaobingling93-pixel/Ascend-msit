
# 背景

msmodelsim权重格式与开源工具AutoAWQ以及AutoGPTQ的格式存在差异，因此本文的目的是提供一份指南，用于将msmodeslim量化后的权重转换为与如上的开源工具格式一致的权重，以实现qwen2-7b W4A16转换后的权重能直接以hugingface形式加载权重。
其中AutoAWQ仅支持w4a16的pergroup量化。AutoGPTQ支持W4A16和W8A16的per group和per channel量化。转换流程如下：


# 1.npu量化以及转换
## 1.npu量化
量化脚本跟正常的量化脚本一样，需要注意的地方有两处：
a.离群值抑制AntiOutlier  awq需要修改，量化方式以及anti_method="m3"，表示使用awq算法，gptq不需要离群值抑制模块，注释即可。
```python
    anti_config = AntiOutlierConfig(anti_method="m3", dev_type="npu", a_bit=16, w_bit=4, dev_id=device_id，w_sym=True)
    anti_outlier = AntiOutlier(model, calib_data=dataset_calib, cfg=anti_config)
    anti_outlier.process()
```

b.QuantConfig配置
perchannel和pergroup的参数配置是有差异的。
1)pergroup需要配置这三个参数：is_lowbit=True, open_outlier=False, group_size=128。 per channel。
2)如果是AutoGPTQ需要更改w_method为='GPTQ', 如下的三个参数不需要配置，注释掉：is_lowbit=True, open_outlier=False, group_size=128。另外开启gptq跑量化时间相对较长。
如下为AutoAWQ的pergroup配置：

```python
    quant_config = QuantConfig(
    a_bit=16,
    w_bit=4,
    disable_names=disable_names,
    mm_tensors=False,
    dev_type='npu',
    dev_id=0,
    w_sym=True,
    w_method='MinMax',
    is_lowbit=True,
    open_outlier=False,
    group_size=128
)
```


# 2.转换脚本使用
转换脚本路径：msmodelslim\example\ms_to_vllm.py
本文以w4a16为例子进行讲解。使用量化脚本对权重进行量化，生成quant_weight_description_w4a16.json和quant_model_weight_w4a16.safetensors
使用ms_to_vllm.py进行格式转换，转换后生成一个新的safetensors用法：
```python 
    python ms_to_llm.py --model 量化后的权重safetensors文件 --json 量化后的权重描述json文件 --save_path  转化后的权重safetensors文件保存路径  --w_bit 权重位数   --target_tool 目标工具格式  
    参数说明：
        model和json为为必须输入的值
        save_path有默认值res.safetensors
        w_bit默认值为4 
        target_tool默认的工具为awq
使用示例：
    python ms_to_llm.py --model ./quant_model_weight_w4a16.safetensor  --json ./quant_weight_description_w4a16.json   --save_path res.safetensors --target_tool awq 
```


# 2.awq量化以及推理
开源工具相关的环境配置和代码仓库，参考github, 量化和推理参考如下的readme.md，链接如下：
    awq: https://github.com/casper-hansen/AutoAWQ
    gptq: https://github.com/AutoGPTQ/AutoGPTQ

### 2.1awq量化
awq量化, 需要注意的是，Version使用GEMM，如果没有传入数据集可能会报错，需要传入数据集val.jsonl文件，报错信息可以在github上面的Issues搜索。
awq量化脚本如下：
````python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
from datasets import load_dataset
import torch


model_path = 'qwen2_7b_instruct'
quant_path = 'quant_qwen2_7b_awq_4_g128'
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

# Load model
model = AutoAWQForCausalLM.from_pretrained(
    model_path, fulse_layers=True, use_cache=False, device_map='auto'
    torch_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

data = load_dataset("json", data_files='./val.jsonl')['train']

calib_data = [text for text in data["text"] if text.strip() != '' and len(text.split('  ')) > 20]


# Quantize
model.quantize(tokenizer, quant_config=quant_config, calib_data=calib_data)

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

printf(f'Model is quantized and saved at "{quan_path}"')

```

### 2.2awq推理
量化后在存储的目标文件当中。
修改awq量化后的脚本目录里的model.safetensors.index.json文件, 将经过里面的safetensors名字转换为经过ms量化后的权重文件,比如更改为res.safetenors，最后运行推理脚本.

awq推理脚本测试对话如下：
```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = './qwen2_7b_instruct'
quant_path = './quant_qwen2_7b'
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

# Load model
model = AutoAWQForCausalLM.from_pretrained(model_path, fulse_layers=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True,
                                                use_fast=False)

test_prompt = "what is deep learning:"
test_input = tokenizer(test_prompt, return_tensors="pt")
print("model is inferring...")
model.eval()
generate_ids = model.generate(
    test_input.input_ids.cuda(),
    attention_mask=test_input.attention_mask.cuda()), 
    max_new_tokens=16
)

res = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
for idx, item in enumerate(res):
    print(item)
```


## 2.2gptq量化以及推理
ms转换为gptq进行推理和awq同理。
首先去阅读gptq的readme.md，找到量化部分的示例，修改路径，和相关配置参数，运行即可。

推理脚本
```python
from transformers import AutoTokenizer, TextGenerationPipeline
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

pretrained_model_dir = "./qwen2_7b_instruct"
quantized_model_dir = "./ms_to_gptq"

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)
examples = [
    tokenizer(
        "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
    )
]

# 加载未量化的模型，默认情况下，模型总是会被加载到 CPU 内存中
model = AutoGPTQForCausalLM.from_pretrained(pretrained_model_dir, quantize_config, device="cuda:0")
print(tokenizer.decode(model.generate(**tokenizer("auto_gptq is", return_tensors="pt").to(model.device))[0]))
```

msmodelslim量化后，并且量化权重经过转换脚本转换转换后能够再AutoAWQ和AutoGPTQ推理成功，则说明转换成功，则说明对于该模型，两边量化工具的量化是没有问题的，能够支持npu转换到gpu上运行。









