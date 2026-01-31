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
import re
import subprocess
from glob import glob

from components.utils.file_open_check import FileStat, is_legal_args_path_string
from components.utils.file_utils import check_input_file_path, check_input_dir_path
from components.utils.util import filter_cmd

MAX_SIZE_LIMITE_NORMAL_MODEL = 32 * 1024 * 1024 * 1024  # 32GB


def check_in_path_legality(value):
    path_value = value
    try:
        file_stat = FileStat(path_value)
    except Exception as err:
        raise argparse.ArgumentTypeError("Input path or file is illegal, please check.") from err
    if file_stat.is_file:
        check_input_file_path(path_value)
    if file_stat.is_dir:
        check_input_dir_path(path_value)
    if not file_stat.is_basically_legal('read'):
        raise argparse.ArgumentTypeError("The current input file does not have right read permission, please check.")
    if file_stat.is_file and not file_stat.is_legal_file_type(["onnx"]):
        raise argparse.ArgumentTypeError("The file type of input path is illegal. Only support [.onnx] file.")
    if file_stat.is_file and not file_stat.is_legal_file_size(MAX_SIZE_LIMITE_NORMAL_MODEL):
        raise argparse.ArgumentTypeError("The current software version only supports input files up to 32GB in size.")
    return path_value


def check_in_model_path_legality(value):
    path_value = value
    try:
        file_stat = FileStat(path_value)
    except Exception as err:
        raise argparse.ArgumentTypeError("Input model path is illegal, please check.") from err
    if file_stat.is_file:
        check_input_file_path(path_value)
    if not file_stat.is_basically_legal('read'):
        err_msg = "The current input model file does not have right read permission, please check."
        raise argparse.ArgumentTypeError(err_msg)
    if not file_stat.is_legal_file_type(["onnx"]):
        raise argparse.ArgumentTypeError("The input model type is illegal. Only support [.onnx] model.")
    if not file_stat.is_legal_file_size(MAX_SIZE_LIMITE_NORMAL_MODEL):
        raise argparse.ArgumentTypeError("The current software version only supports input model up to 32GB in size.")
    return path_value


def check_out_model_path_legality(value):
    if not value:
        return None
    path_value = value
    try:
        file_stat = FileStat(path_value)
    except Exception as err:
        raise argparse.ArgumentTypeError("Output model path is illegal, please check.") from err
    if not file_stat.is_basically_legal('write'):
        err_msg = "The current output model path does not have right write permission, please check."
        raise argparse.ArgumentTypeError(err_msg)
    if not file_stat.is_legal_file_type(["onnx"]):
        raise argparse.ArgumentTypeError("The output model type is illegal. Only support [.onnx] model.")
    return path_value


def check_soc(value):
    if isinstance(value, str) and not re.match("^[0-9]+?$", value):
        raise argparse.ArgumentTypeError("The input 'device' param is not valid.")
    ivalue = int(value)
    pre_cmd = ["ls"]
    pre_cmd.extend(glob("/dev/davinci*"))
    if len(pre_cmd) > 1:
        pre_cmd = filter_cmd(pre_cmd)
        process = subprocess.run(pre_cmd, shell=False, stdout=subprocess.PIPE)
        max_device_id = len(process.stdout.decode().split()) - 2
        if ivalue > max_device_id or ivalue < 0:
            raise argparse.ArgumentTypeError(f"{value} is not a valid value. Please check device id. ")
    else:
        raise RuntimeError(
            "No davinci* files in the /dev/ directory. The current device may not have the Ascend NPU suite installed."
        )
    return ivalue


def check_range(value):
    if isinstance(value, str) and not re.match("^[0-9]+?$", value):
        raise argparse.ArgumentTypeError("The input 'processes' param is not valid.")
    ivalue = int(value)
    if ivalue < 1 or ivalue > 64:
        raise argparse.ArgumentTypeError(f"{value} is not a valid value. Range 1 ~ 64.")
    return ivalue


def check_min_num_1(value):
    if isinstance(value, str) and not re.match("^[0-9]+?$", value):
        raise argparse.ArgumentTypeError("The input 'loop' param is not valid.")
    ivalue = int(value)
    if ivalue < 1:
        raise argparse.ArgumentTypeError(f"{value} is not a valid value. Minimum value 1.")
    return ivalue


def check_min_num_2(value):
    if isinstance(value, str) and not re.match("^[-]?[0-9]+?$", value):
        raise argparse.ArgumentTypeError("The input 'threshold' param is not valid.")
    ivalue = int(value)
    if ivalue < -1:
        raise argparse.ArgumentTypeError(f"{value} is not a valid value. Minimum value -1.")
    return ivalue


def check_shapes_string(value):
    if not value:
        return value
    shapes_string = value
    regex = re.compile(r"[^_A-Za-z0-9,;:/.-]")
    if regex.search(shapes_string):
        raise argparse.ArgumentTypeError(f"shapes string \"{shapes_string}\" is not a legal string")
    return shapes_string


def check_dtypes_string(value):
    if not value:
        return value
    dtypes_string = value
    regex = re.compile(r"[^_A-Za-z0-9;:/.-]")
    if regex.search(dtypes_string):
        raise argparse.ArgumentTypeError(f"dtypes string \"{dtypes_string}\" is not a legal string")
    return dtypes_string


def check_io_string(value):
    if not value:
        return value
    io_string = value
    regex = re.compile(r"[^_A-Za-z0-9,;:/.-]")
    if regex.search(io_string):
        raise argparse.ArgumentTypeError(f"io string \"{io_string}\" is not a legal string")
    return io_string


def check_nodes_string(value):
    if not value:
        return value
    nodes_string = value
    regex = re.compile(r"[^_A-Za-z0-9,:/.-]")
    if regex.search(nodes_string):
        raise argparse.ArgumentTypeError(f"nodes string \"{nodes_string}\" is not a legal string")
    return nodes_string


def check_single_node_string(value):
    if not value:
        return value
    node_string = value
    regex = re.compile(r"[^_A-Za-z0-9:/.-]")
    if regex.search(node_string):
        raise argparse.ArgumentTypeError(f"single_node string \"{node_string}\" is not a legal string")
    return node_string


def check_normal_string(value):
    if not value:
        return value
    nor_string = value
    regex = re.compile(r"[^_A-Za-z0-9\"'><=\[\])(,}{: /.~-]")
    if regex.search(nor_string):
        raise argparse.ArgumentTypeError(f"single_node string \"{nor_string}\" is not a legal string")
    return nor_string


def check_shapes_range_string(value):
    if not value:
        return value
    range_string = value
    regex = re.compile(r"[^_A-Za-z0-9,;:/.\-~]")
    if regex.search(range_string):
        raise argparse.ArgumentTypeError(f"dym range string \"{range_string}\" is not a legal string")
    return range_string


def check_ints_string(value):
    if not value:
        return value
    ints_string = value
    regex = re.compile(r"[^0-9,]")
    if regex.search(ints_string):
        raise argparse.ArgumentTypeError(f"ints string \"{ints_string}\" is not a legal string")
    return ints_string


def check_path_string(value):
    if not value:
        return value
    path_string = value
    if not is_legal_args_path_string(path_string):
        raise argparse.ArgumentTypeError(f"ints string \"{path_string}\" is not a legal string")
    return path_string
