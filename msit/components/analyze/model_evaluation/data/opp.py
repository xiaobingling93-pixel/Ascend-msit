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
import json

from typing import Dict

from model_evaluation.common import utils, Const
from components.utils.log import logger
from model_evaluation.common.enum import SocType, Engine
from model_evaluation.bean import OpInnerInfo
from components.utils.check.rule import Rule
from components.utils.file_open_check import ms_open
from components.utils.constants import TENSOR_MAX_SIZE


class Opp:
    def __init__(self, ops_dict: Dict[str, Dict]) -> None:
        self._ops_dict: Dict[str, Dict] = ops_dict

    @classmethod
    def load_opp(cls, soc_type: str, out_path: str) -> 'Opp':
        latest_path = os.getenv('ASCEND_TOOLKIT_HOME')
        opp_path = os.getenv('ASCEND_OPP_PATH')
        if not os.path.isdir(latest_path):
            raise RuntimeError(f'latest path {latest_path} not exist.')
        if not os.path.isdir(opp_path):
            raise RuntimeError(f'opp path {opp_path} not exist.')

        opp_json = os.path.join(out_path, 'opp.json')
        if not utils.check_file_security(opp_json):
            raise RuntimeError(f'check file security error, file:{opp_json}')
        if os.path.isfile(opp_json):
            os.remove(opp_json)

        fast_query_shell = os.path.join(latest_path, Const.FAST_QUERY_BIN)
        exec_cmd = [
            'python3',
            '{}'.format(fast_query_shell),
            '--type=op',
            '--opp_path={}'.format(opp_path),
            '--output={}'.format(opp_json),
        ]
        _, err = utils.exec_command(exec_cmd)
        if len(err) != 0:
            raise RuntimeError(f'exec fast_query shell failed, err:{err}.')

        if not os.path.isfile(opp_json):
            raise RuntimeError(f'{opp_json} not generate.')
        ops_dict = cls.parse_opp_json(opp_json, soc_type)
        if len(ops_dict) == 0:
            raise RuntimeError('parse opp data error.')

        os.remove(opp_json)
        return cls(ops_dict)

    @classmethod
    def parse_opp_json(cls, opp_json: str, soc_type: str) -> Dict[str, Dict]:
        try:
            Rule.input_file().check(opp_json, will_raise=True)
            with ms_open(opp_json, max_size=TENSOR_MAX_SIZE) as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f'load opp json failed, err:{e}')
            return {}

        def handle_soc_type(soc_type):
            if soc_type.startswith(SocType.Ascend310P.name):
                return SocType.Ascend310P.name
            return soc_type

        op_infos = data.get('ops')
        if not isinstance(op_infos, list):
            logger.error('opp data format not match.')
            return {}

        ops_dict: Dict[str, Dict] = {}
        soc_type = handle_soc_type(soc_type)
        for op_info in op_infos:
            if op_info.get('hardware_type') != soc_type:
                continue
            op_type = op_info.get('op_type')
            if isinstance(op_type, str):
                ops_dict[op_type] = op_info
        return ops_dict

    def query_op_info(self, op_type: str) -> OpInnerInfo:
        if op_type not in self._ops_dict:
            return OpInnerInfo()

        return OpInnerInfo(op_type=op_type, op_engine=self.query_op_engine(op_type))

    def query_op_engine(self, op_type: str) -> Engine:
        # get op engine, default AICORE
        op_info = self._ops_dict.get(op_type)
        if op_info is None:
            return Engine.UNKNOWN

        try:
            engine = op_info.get('attr').get('attr').get('opInfo').get('engine')
        except AttributeError:
            # default Aicore
            return Engine.AICORE

        engine_map = {'DNN_VM_AICPU': Engine.AICPU, 'DNN_VM_HOST_CPU': Engine.HOST_CPU, 'DNN_VM_DVPP': Engine.DVPP}

        if engine not in engine_map:
            logger.error(f'engine({engine}) is unknown.')
            return Engine.UNKNOWN
        return engine_map.get(engine)
