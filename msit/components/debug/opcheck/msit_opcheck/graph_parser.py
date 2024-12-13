# -*- coding: utf-8 -*-
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

import json

from components.utils.file_open_check import ms_open

MAX_GE_GRAPH_SIZE = 209715200   # 200 * 1024 * 1024, 200MB


class InputOutputDesc(object):
    def __init__(self, **kwargs):
        """
        kwargs like that:
            "attr": [],
            "device_type": "NPU",
            "dtype": "DT_BOOL",
            "layout": "ND",
            "real_dim_cnt": 1,
            "shape": {
                "dim": [
                    1
                ]
            },
        """
        self.attr = kwargs.get("attr")
        self.input_output_param = kwargs


class OpInfo(object):
    def __init__(self, op_info_dict):
        """
        op_info_dict like that:
            "attr": [],
            "dst_index": [],
            "dst_name": "output",
            "has_out_attr": true,
            "id": 1,
            "input": [],
            "input_desc": [],
            "input_i": [],
            "input_name": [],
            "is_input_const": [],
            "name": "output",
            "output_desc": [],
            "output_i": [],
            "src_index": [],
            "src_name": [],
            "type": "add",
        """
        self.input_desc_list = [InputOutputDesc(**i) for i in op_info_dict.get("input_desc")]
        self.output_desc_list = [InputOutputDesc(**i) for i in op_info_dict.get("output_desc")]
        self.op_keys = op_info_dict.keys()
        self.param = op_info_dict


def parse_ge_op_from_dump_json(json_path, op_type):
    with ms_open(json_path, max_size=MAX_GE_GRAPH_SIZE) as f:
        ge_file = json.load(f)

    graph = ge_file.get("graph")
    op_list = graph[0].get("op")
    final_op = None
    for op in op_list:
        if op.get("type") == op_type:
            final_op = op
    
    op_info = OpInfo(final_op)
    return op_info