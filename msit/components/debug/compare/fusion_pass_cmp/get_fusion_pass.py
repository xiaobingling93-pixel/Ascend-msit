# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
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

import json
import pandas as pd
import numpy as np

from components.utils.log import logger
from components.utils.constants import JSON_FILE_MAX_SIZE
from components.utils.file_open_check import ms_open
from components.utils.check.rule import Rule


def fusion_pass_analysis(csv_result_path, fusion_json_path, out_csv_path, fusion_node_switch):
    op_info_dict = read_fusion_pass_from_json(fusion_json_path)
    if fusion_node_switch:
        write_fusion_to_csv(csv_result_path, op_info_dict, out_csv_path)
    else:
        write_output_to_csv(csv_result_path, op_info_dict, out_csv_path)


def get_single_op_info_from_op_list(op_list):
    op_info_dict = {}
    for op in op_list:
        pass_name = set()
        op_name = op.get("name", None)
        if not op_name:
            continue
        for attr in op['attr']:
            if attr['key'] == 'pass_name':
                pass_name.update(attr['value']['list']['s'])
            if attr['key'] == 'pass_name_ub':
                pass_name.add(attr['value']['s'])
        if pass_name:
            op_info_dict[op_name] = pass_name
    return op_info_dict


def read_fusion_pass_from_json(fusion_json_path):
    try:
        with ms_open(fusion_json_path, max_size=JSON_FILE_MAX_SIZE) as f:
            ge_fusion_data = json.load(f)
    except Exception as e:
        logger.error(f'load json failed, err:{e}')
        return {}

    graph = ge_fusion_data.get("graph")
    op_info_dict = {}
    for sub_graph in graph:
        op_list = sub_graph.get("op")
        sub_info_dict = get_single_op_info_from_op_list(op_list)
        op_info_dict.update(sub_info_dict)
    return op_info_dict


def write_fusion_to_csv(file_path, op_info_dict, out_csv):
    try:
        Rule.input_file().check(file_path, will_raise=True)
        df = pd.read_csv(file_path, keep_default_na=False)
    except Exception as e:
        logger.error(f'load input csv failed, err:{e}')
        return
        
    df = df[df['TensorIndex'].str.contains('output', na=False)]
    df_selected = df.drop(columns=['Index', 'Address.1', 'DataType.1', 'CompareFailReason'])
    filtered_rows = []
    for _, row in df_selected.iterrows():
        if row['NPUDump'] in op_info_dict.keys():
            row['PassName'] = op_info_dict[row['NPUDump']]
            filtered_rows.append(row)
    filtered_df = pd.DataFrame(filtered_rows)
    filtered_df['MatchError'] = np.where(filtered_df['CosineSimilarity'] == 'NaN', "Fusion node not match", "")
    filtered_df.to_csv(out_csv, index=False)
    return


def write_output_to_csv(file_path, op_info_dict, out_csv):
    try:
        Rule.input_file().check(file_path, will_raise=True)
        df = pd.read_csv(file_path, keep_default_na=False)
    except Exception as e:
        logger.error(f'load input csv failed, err:{e}')
        return

    df = df[df['TensorIndex'].str.contains('output', na=False)]
    df['PassName'] = df['NPUDump'].apply(lambda x: op_info_dict.get(x, None))
    df['MatchError'] = np.where(df['CosineSimilarity'] == 'NaN', "Node not match", "")
    df = df.drop(columns=['Index', 'Address.1', 'DataType.1', 'CompareFailReason'])
    df.to_csv(out_csv, index=False)
    return