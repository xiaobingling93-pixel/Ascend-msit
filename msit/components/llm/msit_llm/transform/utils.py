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
import re
import sys
from collections import namedtuple
from pathlib import Path
from dataclasses import dataclass

import torch
from safetensors.torch import safe_open
import torch_npu

from components.utils.util import safe_torch_load
from components.utils.file_open_check import ms_open
from msit_llm.common.log import logger
from msit_llm.common.utils import load_file_to_read_common_check
from components.utils.check.rule import Rule


_SCENARIOS = ["torch_to_float_atb", "float_atb_to_quant_atb", "torch_to_float_python_atb"]
SCENARIOS = namedtuple("SCENARIOS", _SCENARIOS)(*_SCENARIOS)
SOC_VERSION = (100, 101, 102, 103, 104, 200, 201, 202, 203)


@dataclass
class NPUSocInfo:
    soc_name: str = ""
    soc_version: int = -1
    need_nz: bool = False

    def __post_init__(self):
        self.soc_version = torch_npu._C._npu_get_soc_version()
        if self.soc_version in SOC_VERSION:
            self.need_nz = True


def load_atb_speed():
    atb_speed_home_path: str = os.getenv("ATB_SPEED_HOME_PATH", None)
    try:
        Rule.input_dir().check(atb_speed_home_path)
    except Exception as e:
        logger.error(f'Failed to abtain ATB_SPEED_HOME_PATH, err:{e}')
    lib_path = os.path.join(atb_speed_home_path, "lib/libatb_speed_torch.so")
    try:
        Rule.input_file().check(lib_path)
    except Exception as e:
        logger.error(f'Failed to load libatb_speed_torch.so, err:{e}')
    torch.classes.load_library(lib_path)
    sys.path.append(os.path.join(atb_speed_home_path, 'lib'))


def get_transform_scenario(source_path, to_python=False):
    if os.path.isfile(source_path) and source_path.endswith(".cpp"):  # Single cpp input
        return SCENARIOS.float_atb_to_quant_atb

    try:
        from transformers.configuration_utils import PretrainedConfig

        _ = PretrainedConfig.get_config_dict(source_path)
        return SCENARIOS.torch_to_float_python_atb if to_python else SCENARIOS.torch_to_float_atb
    except Exception:
        pass  # Not an error

    if os.path.isdir(source_path) and any([ii.endswith(".cpp") for ii in os.listdir(source_path)]):
        return SCENARIOS.float_atb_to_quant_atb
    else:
        return None


def write_file(save_path, string):
    with ms_open(save_path, 'w') as ff:
        ff.write(string)


def load_model_dict(model_path):
    if Path(model_path).is_file():
        model_path = load_file_to_read_common_check(model_path)
        state_dict = safe_torch_load(model_path)
        return state_dict
    elif Path(model_path).is_dir():
        suffix_list = ['.bin', '.safetensors', '.pt']
        for suffix in suffix_list:
            file_list = list(Path(model_path).glob('*' + suffix))
            if not file_list:
                continue
            state_dict = {}
            for fp in file_list:
                fp = load_file_to_read_common_check(str(fp))

                if suffix == '.safetensors':
                    with safe_open(fp, framework='pt') as ff:
                        ss = {kk: ff.get_tensor(kk).half() for kk in ff.keys()}
                else:
                    ss = safe_torch_load(fp)

                state_dict.update(ss)
            return state_dict
    return {}


def check_if_safe_string(str_value):
    if not re.search(r'^[a-zA-Z0-9_./-]+$', str_value):
        raise ValueError("String parameter contains invalid characters.")
