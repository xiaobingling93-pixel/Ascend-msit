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
"""
Function:
This module is used to read and convert dump data for external users. 
When you use it, you need to import the following function.
"""

import os
import sys

import numpy as np
import torch

from components.utils.log import logger
from components.llm.msit_llm.common.utils import check_output_path_legality
from components.llm.msit_llm.common.tool import TensorBinFile
from components.utils.check.rule import Rule
from components.utils.util import load_file_to_read_common_check


DEFAULT_PARSE_DTYPE = "uint8"


def read_bin_data(bin_data_path):
    if bin_data_path.endswith(".bin"):
        bin_data_path = load_file_to_read_common_check(bin_data_path)
        bin_tensor = TensorBinFile(bin_data_path)
        return bin_tensor

    raise ValueError(f"{bin_data_path} must be end with .bin.")


def convert_bin_data_to_pt(bin_tensor):
    data = bin_tensor.get_data()
    return data


def get_bin_data_from_dir(dump_data_dir, max_depth=20):
    dump_data_dir = os.path.realpath(dump_data_dir)
    Rule.input_dir().check(dump_data_dir)
    all_dump_real_path_list = []
    dump_data_dir_len = len(dump_data_dir.split('/'))
    for root, _, files in os.walk(dump_data_dir):
        root_len = len(root.split('/'))
        if root_len - dump_data_dir_len >= max_depth:
            logger.error("Parse of bin dump data depth exceeds the max recursion limit %d." % (max_depth))
            raise RecursionError("Maximum recursion depth exceeded in comparison.")
        for filename in files:
            if filename.endswith(".bin"):
                bin_dump_file_real_path = os.path.join(root, filename)
                bin_dump_file_real_path = load_file_to_read_common_check(bin_dump_file_real_path)
                all_dump_real_path_list.append(bin_dump_file_real_path)
    return all_dump_real_path_list


def read_dump_data(dump_data_path):
    cann_path = os.environ.get("TOOLCHAIN_HOME", os.environ.get("ASCEND_TOOLKIT_HOME", ""))
    sys.path.append(os.path.join(cann_path, "tools", "operator_cmp", "compare"))

    from dump_parse.dump_utils import parse_dump_file  # Parser tool from CANN msaccucmp
    from cmp_utils.constant.const_manager import ConstManager
    dump_data_path = load_file_to_read_common_check(dump_data_path)
    bin_dump_data = parse_dump_file(dump_data_path, dump_version=ConstManager.OLD_DUMP_TYPE)
    return bin_dump_data


def convert_bin_data_to_npy(bin_dump_data, dtype=DEFAULT_PARSE_DTYPE):
    inputs = [np.frombuffer(input_data.data, dtype=dtype) for input_data in bin_dump_data.input_data]
    outputs = [np.frombuffer(output_data.data, dtype=dtype) for output_data in bin_dump_data.output_data]
    return inputs, outputs

    
def save_torch_data(pt_data, pt_file_path):
    if not pt_file_path.endswith((".pt", ".pth")):
        raise ValueError("Torch file path must be end with .pt or .pth.")
    if not os.path.exists(os.path.dirname(pt_file_path)):
        os.makedirs(os.path.dirname(pt_file_path))
    check_output_path_legality(pt_file_path)
    torch.save(pt_data, pt_file_path)


def save_npy_data(npy_file_path, npy_data):
    if not npy_file_path.endswith(".npy"):
        raise ValueError("Numpy file path must be end with .npy.")
    if not os.path.exists(os.path.dirname(npy_file_path)):
        os.makedirs(os.path.dirname(npy_file_path))
    check_output_path_legality(npy_file_path)
    np.save(npy_file_path, npy_data)