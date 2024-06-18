#!/usr/bin/env python
# coding=utf-8
# Copyright (c) 2023-2023 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Function:
This class mainly involves the main function.
"""
import re
import argparse
import sys
import os

from msquickcmp.cmp_process import cmp_process
from msquickcmp.adapter_cli.args_adapter import CmpArgsAdapter
from msquickcmp.common import utils
from msquickcmp.common.args_check import (
    check_model_path_legality, check_om_path_legality, check_weight_path_legality, check_input_path_legality,
    check_cann_path_legality, check_output_path_legality, check_dict_kind_string, check_device_range_valid,
    check_number_list, check_dym_range_string, check_fusion_cfg_path_legality, check_quant_json_path_legality,
    safe_string, str2bool
)


def _accuracy_compare_parser(compare_parser):
    compare_parser.add_argument("-m", "--model-path", dest="model_path", default="",
                        type=check_model_path_legality,
                        help="<Required> The original model (.onnx or .pb or .prototxt) file path", required=True)
    compare_parser.add_argument("-om", "--offline-model-path", dest="offline_model_path", default="",
                        type=check_om_path_legality,
                        help="<Required> The offline model (.om) file path", required=True)
    compare_parser.add_argument("-i", "--input-path", dest="input_path", default="",
                        type=check_input_path_legality,
                        help="<Optional> The input data path of the model."
                             " Separate multiple inputs with commas(,). E.g: input_0.bin,input_1.bin")
    compare_parser.add_argument("-c", "--cann-path", dest="cann_path",
                        type=check_cann_path_legality,
                        default="/usr/local/Ascend/ascend-toolkit/latest/",
                        help="<Optional> The CANN installation path")
    compare_parser.add_argument("-o", "--out-path", dest="out_path", default="",
                        type=check_output_path_legality, help="<Optional> The output path")
    compare_parser.add_argument("-s", "--input-shape", dest="input_shape", default="",
                        type=check_dict_kind_string,
                        help="<Optional> Shape of input shape. Separate multiple nodes with semicolons(;)."
                             " E.g: input_name1:1,224,224,3;input_name2:3,300")
    compare_parser.add_argument("-d", "--device", dest="device", default="0",
                        type=check_device_range_valid,
                        help="<Optional> Input device ID [0, 255].")
    compare_parser.add_argument("--output-size", dest="output_size", default="",
                        type=check_number_list,
                        help="<Optional> The size of output. Separate multiple sizes with commas(,)."
                             " E.g: 10200,34000")
    compare_parser.add_argument("--output-nodes", dest="output_nodes", default="",
                        type=check_dict_kind_string,
                        help="<Optional> Output nodes designated by user. Separate multiple nodes with semicolons(;)."
                             " E.g: node_name1:0;node_name2:1;node_name3:0")
    compare_parser.add_argument("--advisor", dest="advisor", action="store_true",
                        help="<Optional> Enable advisor after compare.")
    compare_parser.add_argument("-dr", "--dymShape-range", dest="dym_shape_range", default="",
                        type=check_dym_range_string,
                        help="<Optional> Dynamic shape range using in dynamic model, "
                             "using this means ignore input_shape")
    compare_parser.add_argument("--dump", dest="dump", default=True, type=str2bool,
                        help="<Optional> Whether to dump all the operations' ouput.")
    compare_parser.add_argument("--convert", dest="bin2npy", default=False, type=str2bool,
                        help="<Optional> Enable npu dump data conversion from bin to npy after compare.\
                        For example: --convert True")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    _accuracy_compare_parser(parser)
    args = parser.parse_args(sys.argv[1:])

    args.weight_path = None
    cmp_args = CmpArgsAdapter(args.model_path, args.offline_model_path, args.weight_path, args.input_path,
                              args.cann_path, args.out_path, args.input_shape,
                              args.device, args.output_size, args.output_nodes, args.advisor,
                              args.dym_shape_range, args.dump, args.bin2npy)
    try:
        cmp_process(cmp_args, False)
    except utils.AccuracyCompareException as error:
        sys.exit(error.error_info)
