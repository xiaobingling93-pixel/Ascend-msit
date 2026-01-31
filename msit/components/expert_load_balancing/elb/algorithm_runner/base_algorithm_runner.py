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
from enum import Enum
import json

from components.utils.file_open_check import ms_open
from components.utils.security_check import get_valid_write_path


class AlgorithmType(Enum):
    C2LB = 0
    SPECULATIVE_MOE_LEVEL_1 = 1
    DYNAMIC_C2LB = 2
    SPECULATIVE_MOE_LEVEL_2 = 3
    SPECULATIVE_MOE_LEVEL_1_MIXED = 4
    SPECULATIVE_MOE_LEVEL_2_MIXED = 5
    UNKNOWN_TYPE = 999


DEPLOYMENT_JSON_FILE = "{}_global_deployment.json"


class BaseAlgorithmRunner:
    def __init__(self, args):
        self.algorithm_type = AlgorithmType.UNKNOWN_TYPE
        self.args = args
    
    @staticmethod
    def save_json(data, output_path, default=None):
        output_path = get_valid_write_path(output_path)
        with ms_open(output_path, "w") as json_file:
            json.dump(data, json_file, indent=4, default=default)

    def run_algorithm(self, data):
        pass
            