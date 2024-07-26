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
import csv
import torch
import torch.nn as nn 
from msit_llm.common.log import logger
from msit_llm.transform.model_parser import kind, parser
from msit_llm.transform import torch_to_float_atb
from msit_llm.transform.torch_to_float_atb.utils import (get_repeat_box_layer, 
    dag_to_model, init_save_name, init_save_dir)


SMALL_NUM_HIDDEN_LAYERS = 4


def try_setting_small_num_hidden_layers(config):
    if hasattr(config, "num_hidden_layers"):
        config.num_hidden_layers = SMALL_NUM_HIDDEN_LAYERS
    elif hasattr(config, "num_layers"):
        config.num_layers = SMALL_NUM_HIDDEN_LAYERS
    elif hasattr(config, "n_layers"):
        config.n_layers = SMALL_NUM_HIDDEN_LAYERS
    return config
    

def build_model(source_path):
    try:
        from transformers import AutoConfig, AutoModelForCausalLM
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError("transformers package not found, try pip install transformers") from error

    try:
        config = AutoConfig.from_pretrained(source_path, trust_remote_code=True)
        config = try_setting_small_num_hidden_layers(config)
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    except Exception as error:
        raise ValueError(f"build model from {source_path} failed, make sure it works within transformers") from error
    return model


def transform_report(source_path, save_name=None, save_dir=None, is_repeat=True):
    model = build_model(source_path)
    model_layers = get_repeat_box_layer(model)
    from ascend_utils.pytorch.dag.dag_torch_hook import DagTorchHook
    dag_node = DagTorchHook(model, torch.ones([1, 32]).long(), collapse_repeat_block=is_repeat)
    parsed_model_layers = dag_to_model(dag_node, is_repeat, model_layers) 
    parsed_model_layers = {"name": kind.mname(model), "children": parsed_model_layers}

    model_name_lower = parsed_model_layers.get("name", "model").lower()
    json_save_name = init_save_name(save_name if save_name else model_name_lower) + ".json"
    json_save_dir = init_save_dir(save_dir if save_dir else model_name_lower, sub_dir="")
    json_save_path = os.path.join(json_save_dir, json_save_name)
    with open(json_save_path, "w") as ff:
        json.dump(parsed_model_layers, ff)
    logger.info(f"model info saved: {json_save_path}")

    headers = ["ori_op_name", "ori_op_type", "op_name", "op_type", "soc_type", "engine", "is_supported"]
    csv_save_name = init_save_name(save_name if save_name else model_name_lower) + "_operators.csv"
    csv_save_dir = init_save_dir(save_dir if save_dir else model_name_lower, sub_dir="")
    csv_file_path = os.path.join(csv_save_dir, csv_save_name)
    data = []
    ops_dict = {
        'relu': 'ActivationOperation', 
        'gelu': 'ActivationOperation', 
        'hardswish': 'ActivationOperation', 
        'LogSigmoid': 'ActivationOperation', 
        'as_strided': 'AsStridedOperation', 
        'all_gather': 'AllGatherOperation', 
        'all_reduce': 'AllReduceOperation', 
        'full': 'FillOperation', 
        'masked_fill': 'FillOperation', 
        'masked_fill_': 'FillOperation', 
        'transpose': 'TransposeOperation', 
        'cat': 'ConcatOperation', 
        'cumsum': 'CumsumOperation', 
        'matmul': 'MatmulOperation', 
        'gather': 'GatherOperation', 
        'LayerNorm': 'LayerNormOperation', 
        'RMSNorm': 'RmsNormOperation', 
        'Linear': 'LinearOperation', 
        'DistributedDataParallel': 'LinearParallelOperation', 
        'multinomial': 'MultinomialOperation', 
        'index_select': 'SliceOperation', 
        'split': 'SplitOperation', 
        'chunk': 'SplitOperation', 
        'softmax': 'SoftmaxOperation', 
        'repeat': 'RepeatOperation', 
        'where': 'WhereOperation', 
        'broadcast': 'BroadcastOperation', 
        'topk': 'TopkToppSamplingOperation',  
        'Embedding': 'WordEmbedding', 
        'LlamaRMSNorm': 'RmsNormOperation'
    }

    for node in dag_node.dag_node_list:
        if "forward" in node.name:
            continue

        ori_op_name = f"{node.name_in_network}"
        ori_op_type = f"{node.op_type}"
        op_name = None
        op_type = ops_dict.get(ori_op_type, None)
        soc_type = 'Ascend910'
        engine = 'UNKNOWN' if op_type is None else 'AICORE'
        is_supported = 'FALSE' if op_type is None else 'TRUE'

        data.append((ori_op_name, ori_op_type, op_name, op_type, soc_type, engine, is_supported))

    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        for row in data:
            writer.writerow(row)
    logger.info(f"csv info saved: {csv_file_path}")


def transform_float(source_path, save_name=None, save_dir=None):
    logger.info("Building model using transformers...")

    model = build_model(source_path)

    logger.info("Transforming to atb")

    parsed_model = parser.build_model_tree(model)
    model_name_lower = parsed_model.get("name", "model").lower()
    json_save_name = init_save_name(save_name if save_name else model_name_lower) + ".json"
    json_save_dir = init_save_dir(save_dir if save_dir else model_name_lower, sub_dir="")
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