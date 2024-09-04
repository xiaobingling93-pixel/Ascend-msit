# coding=utf-8
# Copyright (c) 2024-2025 Huawei Technologies Co., Ltd.
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
This class mainly dump model ops inputs and outputs.
"""

import os
import time

import acl
from msquickcmp.adapter_cli.args_adapter import DumpArgsAdapter

from components.debug.compare.msquickcmp.atc import atc_utils
from components.debug.compare.msquickcmp.common import utils
from components.debug.compare.msquickcmp.common.args_check import is_saved_model_valid
from components.debug.compare.msquickcmp.common.convert import convert_npy_to_bin
from components.debug.compare.msquickcmp.common.utils import AccuracyCompareException, get_shape_to_directory_name
from components.debug.compare.msquickcmp.npu.npu_dump_data import NpuDumpData, DynamicInput
from components.debug.compare.msquickcmp.npu.om_parser import OmParser
from components.debug.compare.msquickcmp.single_op import single_op as sp
from components.debug.compare.msquickcmp.common.convert import convert_bin_dump_data_to_npy


def _generate_golden_data_model(args: DumpArgsAdapter, npu_dump_npy_path):
    if is_saved_model_valid(args.model_path):
        from components.debug.compare.msquickcmp.tf.tf_save_model_dump_data import TfSaveModelDumpData
        return TfSaveModelDumpData(args, args.model_path)
    model_name, extension = utils.get_model_name_and_extension(args.model_path)
    if args.weight_path and extension == ".prototxt":
        from components.debug.compare.msquickcmp.caffe_model.caffe_dump_data import CaffeDumpData
        return CaffeDumpData(args)
    elif extension == ".pb":
        from components.debug.compare.msquickcmp.tf.tf_dump_data import TfDumpData
        return TfDumpData(args)
    elif extension == ".onnx":
        from components.debug.compare.msquickcmp.onnx_model.onnx_dump_data import OnnxDumpData
        return OnnxDumpData(args, npu_dump_npy_path)
    else:
        utils.logger.error("cpu dump model files whose names end with .pb or .onnx or .prototxt or saved_model are "
                           "supported")
        raise AccuracyCompareException(utils.ACCURACY_COMPARISON_MODEL_TYPE_ERROR)


def _generate_model_adapter(args: DumpArgsAdapter):
    if is_saved_model_valid(args.model_path):
        from components.debug.compare.msquickcmp.npu.npu_tf_adapter_dump_data import NpuTfAdapterDumpData
        return NpuTfAdapterDumpData(args, args.model_path)
    # get model name suffix
    _, extension = utils.get_model_name_and_extension(args.model_path)
    if extension == ".om":
        return NpuDumpData(arguments=args, is_golden=True)


def dump_process(args: DumpArgsAdapter, use_cli: bool):
    """
    Function Description:
        main process function
    Exception Description:
        exit the program when an AccuracyCompare Exception  occurs
    """
    args.model_path = os.path.realpath(args.model)
    args.weight_path = os.path.realpath(args.weight_path) if args.weight_path else None
    args.cann_path = os.path.realpath(args.cann_path)
    args.input_path = convert_npy_to_bin(args.input_path)
    check_and_dump(args, use_cli)


def dump_data(args: DumpArgsAdapter, input_shape, original_out_path, use_cli: bool):
    if input_shape:
        args.input_shape = input_shape
        args.out_path = os.path.join(original_out_path, get_shape_to_directory_name(args.input_shape))
    if args.device_pattern == "npu":
        """
        npu dump
        """
        npu_dump_process(args, use_cli)
    else:
        """
        cpu dump
        """
        cpu_dump_process(args)


def npu_dump_process(args, use_cli):
    # 1. get dumper
    npu_dumper = _generate_model_adapter(args)
    use_aipp = False
    if os.path.splitext(os.path.basename(args.model_path)) == ".om":
        # whether use aipp
        output_json_path = atc_utils.convert_model_to_json(args.cann_path, args.model_path, args.out_path)
        temp_om_parser = OmParser(output_json_path)
        use_aipp = True if temp_om_parser.get_aipp_config_content() else False
        if use_aipp and args.fusion_switch_file is not None:
            utils.logger.error("if .om model is using aipp config, --fusion-switch-file arg is not support.")
            raise AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_PARAM_ERROR)
    # 2. generate input
    npu_dumper.generate_inputs_data(use_aipp=use_aipp)
    # 3. dump data
    npu_dumper.generate_dump_data(use_cli=use_cli)


def cpu_dump_process(args: DumpArgsAdapter):
    # 1. get dumper
    golden_dumper = _generate_golden_data_model(args, npu_dump_npy_path="")
    if is_saved_model_valid(args.model_path):
        # 2. generate input
        golden_dumper.generate_inputs_data_for_dump()
        # 3. dump data
        golden_dumper.generate_dump_data(args.tf_ops_json_path, npu_dump_path=None, om_parser=None)
    else:
        _, extension = utils.get_model_name_and_extension(args.model_path)
        use_aipp = get_use_aipp(args)
        # when onnx cpu inference and add use_aipp, onnx need npu_dump_data_path and npu_net_output_data_path
        if extension == ".onnx" and use_aipp is True:
            if args.bin2npy or args.custom_op != "":
                npu_dump_npy_path = convert_bin_dump_data_to_npy(args.use_aipp_npu_dump_data_path,
                                                                 args.use_aipp_npu_net_output_data_path,
                                                                 args.cann_path)
            else:
                npu_dump_npy_path = ""
            golden_dumper = _generate_golden_data_model(args, npu_dump_npy_path)

        golden_dumper.generate_inputs_data(args.onnx_npu_dump_data_path, use_aipp)
        golden_dumper.generate_dump_data()


def get_use_aipp(args: DumpArgsAdapter):
    temp_om_parser = OmParser(args.om_json_path)
    use_aipp = True if temp_om_parser.get_aipp_config_content() else False
    if use_aipp and args.fusion_switch_file is not None:
        utils.logger.error("if .om model is using aipp config, --fusion-switch-file arg is not support.")
        raise AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_PARAM_ERROR)
    return use_aipp


def fusion_close_model_convert(args: DumpArgsAdapter):
    if args.fusion_switch_file:
        args.fusion_switch_file = os.path.realpath(args.fusion_switch_file)
        utils.check_file_or_directory_path(args.fusion_switch_file)

        om_json_path = atc_utils.convert_model_to_json(args.cann_path, args.offline_model_path, args.out_path)
        om_parser = OmParser(om_json_path)
        atc_input_shape_in_offline_model = DynamicInput.get_input_shape_from_om(om_parser)

        close_fusion_om_file = os.path.join(args.out_path, 'close_fusion_om_model')
        atc_command_file_path = atc_utils.get_atc_path(args.cann_path)
        atc_cmd = [atc_command_file_path, "--framework=5",
                   "--soc_version=" + acl.get_soc_name(),
                   "--model=" + args.model_path,
                   "--output=" + close_fusion_om_file,
                   "--fusion_switch_file=" + args.fusion_switch_file]
        if atc_input_shape_in_offline_model:
            atc_cmd.append("--input_shape=" + atc_input_shape_in_offline_model)

        utils.execute_command(atc_cmd)
        args.model_path = close_fusion_om_file + ".om"


def check_and_dump(args: DumpArgsAdapter, use_cli: bool):
    utils.check_file_or_directory_path(args.model_path)
    if args.weight_path:
        utils.check_file_or_directory_path(args.weight_path)
    utils.check_device_param_valid(args.device)
    utils.check_file_or_directory_path(os.path.realpath(args.out_path), True)
    utils.check_convert_is_valid_used(args.dump, args.bin2npy, args.custom_op)
    utils.check_locat_is_valid(args.dump, args.locat)
    sp.check_single_op_is_valid(args.single_op, args.dump, args.custom_op, args.locat)
    utils.check_max_size_param_valid(args.max_cmp_size)
    time_dir = time.strftime("%Y%m%d%H%M%S", time.localtime())
    original_out_path = os.path.realpath(os.path.join(args.out_path, time_dir))
    args.out_path = original_out_path
    fusion_close_model_convert(args)
    # deal with the dymShape_range param if exists
    input_shapes = []
    if args.dym_shape_range:
        input_shapes = utils.parse_dym_shape_range(args.dym_shape_range)
    if not input_shapes:
        input_shapes.append("")
    for input_shape in input_shapes:
        dump_data(args, input_shape, original_out_path, use_cli)
