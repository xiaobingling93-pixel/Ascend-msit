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
import argparse
import os
import subprocess


from model_convert.aoe.aoe_args_map import aoe_args
from model_convert.atc.atc_args_map import atc_args
from components.utils.util import filter_cmd
from components.utils.security_check import get_valid_read_path, get_valid_write_path, MAX_READ_FILE_SIZE_32G
from components.utils.log import logger

input_file_args = ['model', 'weight', 'singleop', 'insert_op_conf', 'op_name_map', 'fusion_switch_file',
                   'compression_optimize_conf', 'op_debug_config']
input_dir_args = ['mdl_bank_path', 'op_bank_path', 'debug_dir', 'op_compiler_cache_dir', 'model_path']
output_file_args = ['output', 'json', ]


CUR_PATH = os.path.dirname(os.path.relpath(__file__))

BACKEND_ARGS_MAPPING = {
    "atc": atc_args,
    "aoe": aoe_args
}
BACKEND_CMD_MAPPING = {
    "atc": ["atc"],
    "aoe": ["aoe"]
}


def add_arguments(parser, backend="atc"):
    args = BACKEND_ARGS_MAPPING.get(backend)
    if not args:
        raise ValueError("Backend must be atc or aoe!")

    for arg in args:
        abbr_name = arg.get('abbr_name') if arg.get('abbr_name') else ""
        is_required = arg.get('is_required') if arg.get('is_required') else False

        if abbr_name:
            parser.add_argument(abbr_name, arg.get('name'), required=is_required, help=arg.get('desc'))
        else:
            parser.add_argument(arg.get('name'), required=is_required, help=arg.get('desc'))

    return args


def gen_convert_cmd(conf_args: list, parse_args: argparse.Namespace, backend: str = "atc"):
    cmds = BACKEND_CMD_MAPPING.get(backend)
    if not cmds:
        raise ValueError("Backend must be atc or aoe!")

    for arg in conf_args:
        arg_name = arg.get("name")[2:]
        if hasattr(parse_args, arg_name) and getattr(parse_args, arg_name):
            arg_value = str(getattr(parse_args, arg_name))
            if arg_name in input_file_args:
                arg_value = get_valid_read_path(arg_value, size_max=MAX_READ_FILE_SIZE_32G)
            if arg_name in input_dir_args:
                arg_value = get_valid_read_path(arg_value, is_dir=True)
            if arg_name in output_file_args:
                arg_value = get_valid_write_path(arg_value)

            cmds.append(arg.get("name") + "=" + arg_value)

    return cmds


def execute_cmd(cmd: list):
    cmd = filter_cmd(cmd)
    logger.info("%s start converting now", cmd[0].upper())
    result = subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        while result.poll() is None:
            line = result.stdout.readline()
            if line:
                line = line.strip()
                logger.info(line.decode('utf-8'))
    finally:
        result.stdout.close()
    logger.info("%s convert success", cmd[0].upper())
    return result.returncode
