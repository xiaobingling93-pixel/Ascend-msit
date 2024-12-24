
# 使用说明
msModelslim权重格式与开源工具AutoAWQ、AutoGPTQ的格式存在差异，因此本文的目的是提供一份指南，用于将msModelslim量化后的权重转换为与如上的开源工具格式一致的权重，以实现qwen2-7b W4A16转换后的权重能直接以hugingface形式加载权重。
本指南仅支持如下配置的权重转换：  
W4A16 + pergroup + AWQ  
W4A16 + pergroup + GPTQ  
W4A16 + perchannel + GPTQ  
W8A16 + pergroup + GPTQ  
W8A16 + perchannel + GPTQ

使用平台：  
msModelSlim量化：NPU  
转换脚本：CPU  
AutoAWQ：GPU  
AutoGPTQ：GPU


# 1.msModelslim量化
环境准备如下：  
[大模型量化工具使用前的开发环境的部署](https://gitee.com/ascend/msit/tree/master/msmodelslim)  
[大模型量化工具依赖安装](https://gitee.com/ascend/msit/tree/master/msmodelslim/msmodelslim/pytorch/llm_ptq)  

## 1.1 msModelslim量化
量化脚本跟正常的量化脚本一样，可以参考：https://gitee.com/ascend/msit/blob/master/msmodelslim/docs/w8a8%E7%B2%BE%E5%BA%A6%E8%B0%83%E4%BC%98%E7%AD%96%E7%95%A5.md。需要注意的地方有三处：
a.离群值抑制AntiOutlierConfig a_bit和b_bit需要为指定的值，anti_method="m3"，表示使用AWQ算法，GPTQ算法不需要离群值抑制模块，注释即可。
```python
anti_config = AntiOutlierConfig(anti_method="m3", dev_type="npu", a_bit=16, w_bit=4, dev_id=device_id，w_sym=True)  
anti_outlier = AntiOutlier(model, calib_data=dataset_calib, cfg=anti_config)
anti_outlier.process()
```

b.QuantConfig配置
perchannel和pergroup的参数配置是有差异的。
1)pergroup需要配置这三个参数：is_lowbit=True, open_outlier=False, group_size=128。 per channel。
2)如果是AutoGPTQ需要更改w_method为='GPTQ', 如下的三个参数不需要配置，注释掉：is_lowbit=True, open_outlier=False, group_size=128。另外开启GPTQ跑量化时间相对较长。
如下为AutoAWQ的pergroup配置：

```python
quant_config = QuantConfig(
    a_bit=16,                      # 激活值量化位数
    w_bit=4,                       # 权重量化位数
    disable_names=disable_names,   # 手动回退的量化层名称
    mm_tensors=False,              # 默认True，表示使用per-tensor量化，False为per-channel量化
    dev_type='npu',                # 量化的工具为NPU
    dev_id=0,                       
    w_sym=True,                    # 对称量化
    w_method='MinMax',             # 权重量化策略
    is_lowbit=True,                # 如下为pergroup场景下的设置，如果是per-channel量化注释掉如下三个参数
    open_outlier=False,
    group_size=128                 
)
```
c.关于保存的权重文件分片的简介
首先本脚本仅支持未切片的safetensors权重转换，所以使用保存文件的时候，不要使用进行分片保存。  
参考链接：https://gitee.com/ascend/msit/blob/dev/msmodelslim/docs/Python-API%E6%8E%A5%E5%8F%A3%E8%AF%B4%E6%98%8E/%E5%A4%A7%E6%A8%A1%E5%9E%8B%E5%8E%8B%E7%BC%A9%E6%8E%A5%E5%8F%A3/%E5%A4%A7%E6%A8%A1%E5%9E%8B%E9%87%8F%E5%8C%96%E6%8E%A5%E5%8F%A3/PyTorch/save().md
```python
calibrator.save(output_path, safetensors_name=None, json_name=None, save_type=None, part_file_size=None)
```


## 1.2 转换脚本使用
转换脚本路径位于本仓库：msit\msmodelslim\example\ms_to_vllm.py。链接：https://gitee.com/ascend/msit/blob/master/msmodelslim/example/ms_to_vllm.py
本文以W4A16为例子进行讲解。经过上一步1.1使用msModelslim对权重进行量化，生成quant_weight_description_w4a16.json和quant_model_weight_w4a16.safetensors
再使用ms_to_vllm.py进行格式转换，生成转换后的safetensors文件，用法如下：
```python 
命令：
python ms_to_llm.py --model {weighted_safetensors_path} --json {weighted_json_path} --save_path  {converted_safetensors_path}  --w_bit {weight_bit}   --target_tool  {target_convert_tool}

说明：
    model，必须参数，用于表示传入量化后的safetensors权重文件，可传入文件的绝对路径和相对路径
    json，必选参数，用于表示传入的量化后json权重描述文件，传入文件的绝对路径和相对路径
    save_path，可选参数，默认值res.safetensors，表示转换后的文件存储于当前目录的位置，路径为./res.safetensors，save_path不支持创建目录，仅支持创建文件
    w_bit，可选参数，默认值为4，表示量化的权重位数，根据量化的权重写4或8
    target_tool，可选参数，默认值为awq，表示转换的目标工具为AutoAWQ，仅支持awq和gptq，另一个参数gptq，表示AutoGPTQ工具

使用示例：
    python ms_to_llm.py --model ./quant_model_weight_w4a16.safetensor  --json ./quant_weight_description_w4a16.json   --save_path res.safetensors --target_tool awq 
```

# 2.开源工具AutoAWQ量化以及推理
## 2.1环境准备
开源工具相关的环境配置、量化和推理参考github上的readme.md，链接如下：
AutoAWQ: https://github.com/casper-hansen/AutoAWQ

## 2.2量化
AutoAWQ量化, 需要注意的是，Version使用GEMM，如果没有传入数据集可能会报错，需要传入数据集val.jsonl文件, 参考网址：https://github.com/casper-hansen/AutoAWQ/issues/506
，数据集获取地址：https://huggingface.co/datasets/mit-han-lab/pile-val-backup/blob/main/val.jsonl.zst。     
AutoAWQ量化脚本如下：

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
from datasets import load_dataset
import torch


model_path = 'qwen2_7b_instruct'            # 浮点模型权重路径
quant_path = 'quant_qwen2_7b_awq_4_g128'    # 浮点模型经过量化后的保存路径

# q_group_size和 msModelSlim量化的group_size对应，保持一致
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }    

# Load model
model = AutoAWQForCausalLM.from_pretrained(
    model_path, low_cpu_mem_usage=True, use_cache=False, device_map='auto'，
    torch_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

data = load_dataset("json", data_files='./val.jsonl')['train']

calib_data = [text for text in data["text"] if text.strip() != '' and len(text.split(' 
')) > 20]


# Quantize
model.quantize(tokenizer, quant_config=quant_config, calib_data=calib_data)

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

printf(f'Model is quantized and saved at "{quan_path}"')

```

## 2.3推理
首先，修改AutoAWQ量化后权重路径的model.safetensors.index.json文件, 将文件里面的weight_map中的权重文件名称改为msModelSlim转换后的文件名，比如：
"res.safetensors"。最后运行推理脚本。

AutoAWQ推理脚本测试对话如下：
```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

quant_path = './quant_qwen2_7b'         # 浮点模型经过量化后的保存路径

# q_group_size和msModelSlim量化的group_size对应，保持一致
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

# Load model
model = AutoAWQForCausalLM.from_pretrained(quant_path, fulse_layers=True)
tokenizer = AutoTokenizer.from_pretrained(quant_path, trust_remote_code=True, local_files_only=True,
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


# 3 开源工具AutoGPTQ量化以及推理

## 3.1 量化
ms转换为AutoGPTQ进行推理和AutoAWQ同理。参考链接：AutoGPTQ: https://github.com/AutoGPTQ/AutoGPTQ
首先去阅读AutoGPTQ的readme.md，找到量化部分的示例，修改路径，和相关配置参数，运行即可，最后生成量化权重文件。


## 3.2推理
将经过msmodelslim量化以及经过转换脚本转换后的res.safetensors文件传入GPU生成的量化权重目录，替换掉之前的量化权重文件。
推理脚本如下：
```python
from transformers import AutoTokenizer, TextGenerationPipeline
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

model_path = "./qwen2_7b_instruct"      # 浮点模型权重路径
quant_path = "./ms_to_gptq"             # 浮点模型经过量化后的保存路径

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
examples = [
    tokenizer(
        "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
    )
]

# 加载未量化的模型，默认情况下，模型总是会被加载到 CPU 内存中
model = AutoGPTQForCausalLM.from_pretrained(quant_path, device="cuda:0")
print(tokenizer.decode(model.generate(**tokenizer("auto_gptq is", return_tensors="pt").to(model.device))[0]))
```


# 4.总结
经过上述步骤，成功完成NPU的量化，并且量化权重经过转换脚本转换转换后能够在AutoAWQ和AutoGPTQ推理成功。









