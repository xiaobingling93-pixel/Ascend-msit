# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
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

import pandas as pd

from components.expert_load_balancing.elb.data_loader.base_loader import DataType, BaseDataLoader
from components.utils.log import logger
from components.utils.security_check import get_valid_read_path


class VllmTensorLoader(BaseDataLoader):
    def __init__(self, input_args):
        super().__init__(input_args)

    @staticmethod
    def check_input_path(input_path):
        if not os.path.isfile(input_path):
            return None
        
        if input_path.endswith(".pt"):
            return {"decode": input_path}
        return None
        
    def load(self, target_files):
        import torch
        target_file = target_files.get("decode")
        target_file = get_valid_read_path(target_file)
        try:
            target_data = torch.load(target_file, weight_only=True, map_location="cpu")
        except Exception as e:
            raise ValueError("Loading from pt file failed, please check input file path,") from e
        target_data = target_data.numpy()
        if target_data.ndim != 2:
            raise ValueError(f"Loading data from {target_file} failed, "
                             f"data shape should be 2, but got {target_data.ndim}")
        self.process_args(target_data)
        res = {"decode": target_data}
        return res, self.input_args
    
    def process_args(self, data):
        self.input_args.n_layers = data.shape[0]
        self.input_args.n_experts = data.shape[1]
