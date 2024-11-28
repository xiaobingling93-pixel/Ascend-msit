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

import os
import json 
from typing import Optional
import torch 

from components.debug.compare.utils.base_dump_reader import DumpFileReader
from components.utils.file_open_check import ms_open
from components.utils.util import safe_torch_load

DELIMITER_MAP = {"TorchScript": '.', "TorchExport": '_'}
MAX_FILE_READ_SIZE = 100 * 1024   # 100KB


class TorchDumpFileReader(DumpFileReader):
    def __init__(self, cpu_path: str, json_path: str, torch_mode="TorchScript"):
        self.path = cpu_path
        self.json_path = json_path 
        self.torch_mode = torch_mode
        self.key_to_folder = self._map_keys_to_folders()

    def get_tensor(self, key: str) -> torch.Tensor:
        cpu_tensor = None 
        folder_name = self.key_to_folder[key]
        key_with_root = f'root.{folder_name}'
        folder_path = os.path.join(self.path, key_with_root)
        for file_name in os.listdir(folder_path):
            if file_name.startswith('output'):
                key_path = os.path.join(folder_path, file_name)
                cpu_tensor = safe_torch_load(key_path)
                return cpu_tensor
            else:
                continue 
        return cpu_tensor

    def get_keys(self) -> set:
        return set(self.key_to_folder.keys())

    def _filter_keys(self, key_to_fold: dict) -> dict:
        keys = list(key_to_fold.values())
        keys.sort(key=len, reverse=True)
        filtered_keys = set()
        delimiter = DELIMITER_MAP.get(self.torch_mode)
        for key in keys:
            is_contained = any(other_key.startswith(key + delimiter) for other_key in keys if other_key != key)
            if not is_contained:
                filtered_keys.add(key)

        filtered_key_to_folder = {fusion_op: key for fusion_op, key in key_to_fold.items() if key in filtered_keys}

        return filtered_key_to_folder

    def _extract_key_from_jit_node(self, jit_node: str) -> Optional[str]:
        # Here is an example of jit_node: "%968 : Float(1, 64, 56, 56) = aten::relu(%input.9), 
        # scope: __module.layer1/__module.layer1.0/__module.layer1.0.relu # 
        # /home/site-packages/torch/nn/functional.py:1469:0"
        key_none = None
        if len(jit_node.split("scope:")) < 2:
            return key_none
        if "#" in jit_node:
            jit_node_front = jit_node.split("#")[0]
            if jit_node_front:
                jit_node_front = jit_node_front.strip()
                jit_node_front_list = jit_node_front.rsplit("__module.", 1)
                if jit_node_front_list:
                    key_none = jit_node_front_list[-1]
        return key_none

    def _extract_key_from_fx_node(self, jit_node: str) -> Optional[str]:
        #jit_node示例: "/layer1/0/relu/relu_1"
        jit_node = jit_node.split("/")
        if len(jit_node) < 3:
            return None
        jit_node = jit_node[1:-1]
        #模型经过fxGraph->export->compile后，dump下来的downsample算子在映射表中多一个下划线后缀
        cpu_key = ".".join(jit_node).strip('_')
        return cpu_key

    def _extract_key(self, data, key_to_folder: dict, key_to_id: dict):
        for fusion_op, details in data.items():
            id_ = details.get('id', float('inf'))
            jit_node = details.get('jit_node', '')
            if jit_node == '':
                continue
            if self.torch_mode == "TorchScript":
                key = self._extract_key_from_jit_node(jit_node)
            else:
                key = self._extract_key_from_fx_node(jit_node)
            if key is not None:
                key_to_folder[fusion_op] = key
                key_to_id[key] = id_

    def _map_keys_to_folders(self) -> dict:
        key_to_folder = {}
        key_to_id = {}
        json_path = os.path.join(self.json_path, 'op_map_updated.json')

        with ms_open(json_path, max_size=MAX_FILE_READ_SIZE) as f:
            data = json.load(f)
            self._extract_key(data, key_to_folder, key_to_id)

        key_to_folder = self._filter_keys(key_to_folder)
        self.key_to_id = key_to_id

        return key_to_folder

