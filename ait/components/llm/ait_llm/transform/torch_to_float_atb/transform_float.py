# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
from ait_llm.common.log import logger
from ait_llm.transform.model_parser import parser

SMALL_NUM_HIDDEN_LAYERS = 4

def try_setting_small_num_hidden_layers(config):
    if hasattr(config, "num_hidden_layers"):
        config.num_hidden_layers = SMALL_NUM_HIDDEN_LAYERS
    elif hasattr(config, "num_layers"):
        config.num_layers = SMALL_NUM_HIDDEN_LAYERS
    elif hasattr(config, "n_layers"):
        config.n_layers = SMALL_NUM_HIDDEN_LAYERS
    return config
    

def transform_float(source_path, save_name=None, save_dir=None):
    from ait_llm.transform import torch_to_float_atb

    logger.info("Building model using transformers...")

    try:
        from transformers import AutoConfig, AutoModelForCausalLM
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError("transformers package not found, try pip install transformers") from error

    try:
        config = AutoConfig.from_pretrained(source_path, trust_remote_code=True)
        config = try_setting_small_num_hidden_layers(config)
        source_model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    except Exception as error:
        raise ValueError(f"build model from {source_path} failed, make sure it works within transformers") from error

    logger.info("Transforming to atb")

    parsed_model = parser.build_model_tree(source_model)
    model_name_lower = parsed_model.get("name", "model").lower()
    json_save_name = torch_to_float_atb.utils.init_save_name(save_name if save_name else model_name_lower) + ".json"
    json_save_dir = torch_to_float_atb.utils.init_save_dir(save_dir if save_dir else model_name_lower, sub_dir="")
    json_save_path = os.path.join(json_save_dir, json_save_name)
    with open(json_save_path, "w") as ff:
        json.dump(parsed_model, ff)
    logger.info(f"model info saved: {json_save_path}")

    model_cpp_file, _ = torch_to_float_atb.float_model_cpp_gen(parsed_model, save_name=save_name, save_dir=save_dir)
    model_h_file, _ = torch_to_float_atb.float_model_h_gen(parsed_model, save_name=save_name, save_dir=save_dir)
    layer_cpp_file, _ = torch_to_float_atb.float_layer_cpp_gen(parsed_model, save_name=save_name, save_dir=save_dir)
    layer_h_file, _ = torch_to_float_atb.float_layer_h_gen(parsed_model, save_name=save_name, save_dir=save_dir)
    result_files = [model_cpp_file, model_h_file, layer_cpp_file, layer_h_file]
    logger.info("Generated files: [\n    " + ",\n    ".join(result_files) + ",\n]")
    return result_files