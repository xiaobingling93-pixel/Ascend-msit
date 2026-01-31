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
import subprocess
from typing import List

from components.utils.log import logger
from components.utils.util import filter_cmd
from model_evaluation.common.enum import Framework, SocType


def get_soc_type() -> str:
    default_soc = SocType.Ascend310.name
    try:
        import acl

        return acl.get_soc_name()
    except ImportError:
        logger.warning(f'Get soc_version failed, use default {default_soc}.')
    return default_soc


def get_framework(model: str) -> Framework:
    dict_ = {
        # file format and framework map
        '.onnx': Framework.ONNX,
        '.prototxt': Framework.CAFFE,
        '.pb': Framework.TF,
    }
    for suffix, framework in dict_.items():
        if model.endswith(suffix):
            return framework
    return Framework.UNKNOWN


def exec_command(cmd_args: List[str]):
    cmd_args = filter_cmd(cmd_args)
    out = subprocess.run(cmd_args, capture_output=True, shell=False)
    outmsg = out.stdout.decode('utf-8')
    outerr = out.stderr.decode('utf-8')
    return outmsg, outerr


def check_file_security(filepath: str) -> bool:
    if not os.path.exists(filepath):
        return True
    if os.path.islink(filepath):
        logger.error('%r is link.', filepath)
        return False
    if not os.path.isfile(filepath):
        logger.error('%r is not file.', filepath)
        return False
    return True
