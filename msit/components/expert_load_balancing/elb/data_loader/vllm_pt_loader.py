# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------
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
            target_data = torch.load(target_file, weights_only=True, map_location="cpu")
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
