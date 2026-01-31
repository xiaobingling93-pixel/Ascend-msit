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
from enum import unique, Enum
from components.utils.parser import BaseCommand
from components.utils.log import logger
from model_convert.cmd_utils import add_arguments, gen_convert_cmd, execute_cmd


@unique
class SocType(Enum):
    Ascend310 = 0
    Ascend310P = 1


def get_soc_type() -> str:
    default_soc = SocType.Ascend310.name
    try:
        import acl

        return acl.get_soc_name()
    except ImportError:
        logger.warning(f'Get soc_version failed, use default {default_soc}.')
    return default_soc


class ModelConvertCommand(BaseCommand):
    def __init__(self, backend, *args, **kwargs):
        super(ModelConvertCommand, self).__init__(*args, **kwargs)
        self.conf_args = None
        self.backend = backend

    def add_arguments(self, parser, **kwargs):
        self.conf_args = add_arguments(parser, backend=self.backend)

    def handle(self, args, **kwargs):
        convert_cmd = gen_convert_cmd(self.conf_args, args, backend=self.backend)
        execute_cmd(convert_cmd)


def get_cmd_instance():
    atc_cmd = ModelConvertCommand(name="atc", help_info="Convert onnx to om by atc.", backend="atc")
    aoe_cmd = ModelConvertCommand(name="aoe", help_info="Convert onnx to om by aoe.", backend="aoe")
    convert_cmd = BaseCommand(
        name="convert",
        help_info="convert tool converts the model from ONNX, TensorFlow, Caffe and MindSpore to OM. \
                   It supports atc, aoe.",
        children=[atc_cmd, aoe_cmd]
    )
    return convert_cmd
