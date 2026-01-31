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

import torch

from components.utils.file_open_check import ms_open
from msit_llm.common.log import logger
from msit_llm.common.utils import check_output_path_legality, load_file_to_read_common_check
from msit_llm.common.constant import get_ait_dump_path
from components.utils.constants import TENSOR_MAX_SIZE
from components.utils.security_check import ms_makedirs


def dump_data(token_id=-1, data_id=-1, golden_data=None, my_path='', output_path='./'):
    # 传参失败的提示
    if token_id == -1:
        logger.warning('Please check whether token_id passed in are correct')
        return
    elif data_id == -1:
        logger.warning('Please check whether data_id passed in are correct')
        return
    elif not isinstance(golden_data, torch.Tensor):
        logger.warning('The golden_data is not a torch tensor!')
        return
    elif my_path == '':
        logger.warning('Please check whether my_path passed in are correct')
        return
    
    my_path = load_file_to_read_common_check(my_path)
    check_output_path_legality(output_path)

    if golden_data is not None:
        cur_pid = os.getpid()
        device_id = golden_data.get_device()
        output_path_prefix = os.path.join(output_path, get_ait_dump_path(), f"{cur_pid}_{device_id}")
        golden_data_dir = os.path.join(output_path_prefix, "golden_tensor", str(token_id))

        if not os.path.exists(golden_data_dir):
            ms_makedirs(golden_data_dir)

        golden_data_path = os.path.join(golden_data_dir, f'{data_id}_tensor.pth')
        try:
            torch.save(golden_data, golden_data_path)
        except OSError as e:
            logger.error(f"Failed to save tensor to {golden_data_path}: {e}")
            return

        json_path = os.path.join(output_path_prefix, "golden_tensor", "metadata.json")
        write_json_file(data_id, golden_data_path, json_path, token_id, my_path)


def write_json_file(data_id, data_path, json_path, token_id, my_path):
    import json
    if not os.path.exists(json_path):
        json_data = {}
    else:
        json_path = load_file_to_read_common_check(json_path)

        with ms_open(json_path, 'r', max_size=TENSOR_MAX_SIZE) as json_file:
            json_data = json.load(json_file)

    json_data[data_id] = {token_id: [data_path, my_path]}
    with ms_open(json_path, "w") as f:
        json.dump(json_data, f)
