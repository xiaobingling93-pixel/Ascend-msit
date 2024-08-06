# Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
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

import datetime
import os

import torch
import pandas as pd

from safetensors.torch import load_file
from msit_llm.common.log import logger
from msit_llm.common.constant import CSV_CMP_WEIGTH_HEADER
from msit_llm.compare.cmp_utils import save_compare_reault_to_csv, compare_data, set_tensor_basic_info_in_row_data


# 在调用接口那里已经有外部输入路径检查了
def find_safetensors_files(golden_path):
    model_dir_path = os.path.abspath(golden_path)
    # 搜索给定目录下的所有文件，查找并存储safetensors文件路径
    # 如果没有找到safetensors文件就报错
    safetensors_file_list = []
    for dirpath, _, fileweight_ft_keys in os.walk(model_dir_path):
        for file in fileweight_ft_keys:
            if file.endswith('.safetensors'):
                safetensors_file_path = os.path.join(dirpath, file)
                safetensors_file_list.append(safetensors_file_path) 

    if not safetensors_file_list:
        raise FileNotFoundError("No .safetensors files found in the directory.")
    return safetensors_file_list
    

def dequant(weight, weight_offset, weight_scale):
    return (weight - weight_offset) * weight_scale
     

def compare_weight(gp_path, mp_path, output_path):
    try:
        gp_path_list = find_safetensors_files(gp_path)
        mp_path_list = find_safetensors_files(mp_path)
    except FileNotFoundError as e:
        print(e)

    gathered_row_data = []
    sorted_gp_path_list = sorted(gp_path_list)
    mp_dict = load_file(mp_path_list[0])

    for g_path in sorted_gp_path_list:
        gp_dict = load_file(g_path)
        
        for ft_weight_key, ft_weight_value in gp_dict.items():
            if ft_weight_key.endswith('weight'):
                int_weight_value = mp_dict.get(ft_weight_key, None)
                # 比较张量类型
                if ft_weight_value.dtype != int_weight_value.dtype:
                    weight_offset_key = ft_weight_key.replace("weight", "weight_offset")
                    weight_scale_key = ft_weight_key.replace("weight", "weight_scale")
                    weight_offset_value = mp_dict.get(weight_offset_key, None)
                    weight_scale_value = mp_dict.get(weight_scale_key, None)
                    dequant_weight_value = dequant(int_weight_value, weight_offset_value, weight_scale_value)
                    row_data_basic = set_tensor_basic_info_in_row_data(ft_weight_value, dequant_weight_value)
                    row_data = compare_data(ft_weight_value, dequant_weight_value) 
                    row_data.update(row_data_basic)
                    row_data.update({"weight_ft_key":ft_weight_key})
                    gathered_row_data.append(row_data)

    return save_compare_reault_to_csv(gathered_row_data, output_path, columns=CSV_CMP_WEIGTH_HEADER)