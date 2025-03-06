# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import argparse
import functools
import json
import torch
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from msmodelslim.tools.copy_config_files import copy_config_files, modify_config_json
from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlierConfig, AntiOutlier
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig
from msmodelslim.tools.logger import set_logger_level
from msmodelslim.tools.add_safetensors import add_safetensors

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help="The path of float model and tokenizer"),
    parser.add_argument('--save_path', type=str, help="The path to save quant model"),
    parser.add_argument('--layer_count', type=int, default=0, help="Layer count when loading model")
    parser.add_argument('--anti_dataset', type=str, default="./anti_prompt.json",
                        help="The calib data for anti outlier")
    parser.add_argument('--calib_dataset', type=str, default="./calib_prompt.json",
                        help="The calib data for calibration")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size for anti and calibration")
    return parser.parse_args()


def custom_hook(model_config):
    model_config["mla_quantize"] = "w8a8"
    model_config["quantize"] = "w8a8_dynamic"
    model_config["model_type"] = "deepseekv2"


def get_calib_dataset_batch(model_tokenizer, calib_list, batch_size, device="npu"):
    calib_dataset = []
    calib_list = [calib_list[i:i + batch_size] for i in range(0, len(calib_list), batch_size)]
    for calib_data in calib_list:
        inputs = model_tokenizer(calib_data, return_tensors='pt', padding=True).to(device)
        calib_dataset.append(
            [value.to(device) for key, value in inputs.data.items() if isinstance(value, torch.Tensor)])
    return calib_dataset


args = parse_args()
set_logger_level("info")
# 显示整个量化过程各个步骤的进度条
pbar = tqdm(total=5, position=0, desc="Total Process")
model_path = args.model_path
config = AutoConfig.from_pretrained(pretrained_model_name_or_path=model_path, trust_remote_code=True)
config.num_hidden_layers = args.layer_count if args.layer_count != 0 else config.num_hidden_layers
config.model_type = "deepseekv2"

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_path,
                                          config=config,
                                          trust_remote_code=True,
                                          use_fast=True,
                                          add_eos_token=True)

model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_path,
                                             config=config,
                                             trust_remote_code=True,
                                             device_map="auto",
                                             torch_dtype="auto",
                                             max_memory={
                                                 0: "50GiB",
                                                 "cpu": "1500GiB"
                                             },
                                             attn_implementation='eager')

pbar.update(1)

with open(args.anti_dataset, "r") as file:
    anti_prompt = json.load(file)
with open(args.calib_dataset, "r") as file:
    calib_prompt = json.load(file)

anti_dataset = get_calib_dataset_batch(tokenizer, anti_prompt, args.batch_size, model.device)
dataset_calib = get_calib_dataset_batch(tokenizer, calib_prompt, args.batch_size, model.device)

with torch.no_grad():
    anti_config = AntiOutlierConfig(w_bit=8,
                                    a_bit=8,
                                    anti_method='m4',
                                    dev_type='npu',
                                    dev_id=model.device.index)
    anti_outlier = AntiOutlier(model, calib_data=anti_dataset, cfg=anti_config)
    anti_outlier.process()
pbar.update(1)

disable_names = []
for ids in range(config.num_hidden_layers):
    disable_names.append("model.layers." + str(ids) + ".self_attn.kv_b_proj")

quant_config = QuantConfig(
    a_bit=8,
    w_bit=8,
    disable_names=disable_names,
    dev_type='npu',
    dev_id=model.device.index,
    act_method=1,
    pr=1.0,
    w_sym=True,
    mm_tensor=False,
)

calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level="L0")
calibrator.run()
pbar.update(1)

calibrator.save(args.save_path,
                json_name="quant_model_description_w8a8_dynamic.json",
                safetensors_name="quant_model_weight_w8a8_dynamic.safetensors",
                save_type=["safe_tensor"],
                part_file_size=4)

custom_hooks = {
    'config.json': functools.partial(modify_config_json, custom_hook=custom_hook)
}
copy_config_files(input_path=args.model_path, output_path=args.save_path, quant_config=quant_config,
                  custom_hooks=custom_hooks)
pbar.update(1)
add_safetensors(org_paths=args.model_path, target_dir=args.save_path, safetensors_prefix="mtp_float",
               max_file_size_gb=5, prefix="model.layers.61.", quant_type="FLOAT")
pbar.update(1)