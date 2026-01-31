# -*- coding: utf-8 -*-
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
from typing import Dict

import yaml

from model_evaluation.common import utils
from model_evaluation.common.enum import Framework
from components.utils.file_open_check import ms_open
from components.utils.constants import TENSOR_MAX_SIZE


class OpMap:
    def __init__(self, op_map: Dict[str, str]) -> None:
        self._op_map: Dict[str, str] = op_map

    @classmethod
    def load_op_map(cls, framework) -> 'OpMap':
        cur_dir: str = os.path.dirname(os.path.realpath(__file__))
        sub_path: str = os.path.join(cur_dir, 'map_conf')
        if not os.path.isdir(sub_path):
            raise RuntimeError(f'{sub_path} is not dir.')

        framework_yaml_map = {Framework.CAFFE: 'caffe.yaml', Framework.ONNX: 'onnx.yaml', Framework.TF: 'tf.yaml'}
        if framework not in framework_yaml_map:
            raise RuntimeError('framework is invalid.')
        yaml_path = os.path.join(sub_path, framework_yaml_map.get(framework))
        if not os.path.exists(yaml_path):
            raise RuntimeError(f'{yaml_path} not exist.')
        if not utils.check_file_security(yaml_path):
            raise RuntimeError(f'check file security error, file:{yaml_path}')

        try:
            with ms_open(yaml_path, max_size=TENSOR_MAX_SIZE) as f:
                op_map = yaml.safe_load(f)
        except Exception as e:
            raise RuntimeError(f'{e}') from e

        return cls(op_map)

    def map_op(self, ori_op: str) -> str:
        return self._op_map.get(ori_op)

    def map_onnx_op(self, ori_op: str, opset_version: int) -> str:
        key = f'ai.onnx::{str(opset_version)}::{ori_op}'
        return self._op_map.get(key)
