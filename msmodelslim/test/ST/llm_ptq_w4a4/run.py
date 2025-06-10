import os
import json
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from modelslim.pytorch.llm_ptq.anti_outlier import AntiOutlierConfig, AntiOutlier
from modelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig
 
fp16_path = f"{os.environ['PROJECT_PATH']}/resource/llm_ptq/llama3-8b/"
 
config = AutoConfig.from_pretrained(pretrained_model_name_or_path=fp16_path, trust_remote_code=True)
config.num_hidden_layers = 2
config.use_cache = False
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=fp16_path,
    config=config,
    trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=fp16_path,
    config=config,
    torch_dtype='auto',
    device_map='auto',
    trust_remote_code=True
).eval()
 
def get_calib_dataset(tokenizer, calib_list, device="npu"):
    calib_dataset = []
    for calib_data in calib_list:
        inputs = tokenizer(calib_data, return_tensors='pt')
        calib_dataset.append([
            inputs.data['input_ids'].to(device),
            inputs.data['attention_mask'].to(device)
        ])
    return calib_dataset
 
 
calib_set = ["Where is the capital of China?"]
dataset_calib = get_calib_dataset(tokenizer, calib_set, device=model.device)
 
disable_names = []
quant_config = QuantConfig(
    w_bit=4,
    a_bit=4,
    disable_names=[],
    dev_type='npu',  # dev_type="npu", dev_id=0  如果需要使用npu进行量化
    dev_id=model.device.index,
    is_dynamic=True,
)
calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')
calibrator.run()
calibrator.save(f"{os.environ['PROJECT_PATH']}/output/llm_ptq_w4a4", save_type=["numpy", "safe_tensor"])
 
print('Save quant weight success!')