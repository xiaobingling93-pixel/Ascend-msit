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
This class mainly involves the main function.
"""

import os
import time

import acl

from msquickcmp.adapter_cli.args_adapter import DumpArgsAdapter
from components.debug.compare.msquickcmp.atc import atc_utils
from components.debug.compare.msquickcmp.common import utils
from components.debug.compare.msquickcmp.common.args_check import is_saved_model_valid
from components.debug.compare.msquickcmp.common.convert import convert_bin_dump_data_to_npy
from components.debug.compare.msquickcmp.common.convert import convert_npy_to_bin
from components.debug.compare.msquickcmp.common.utils import AccuracyCompareException, get_shape_to_directory_name
from components.debug.compare.msquickcmp.npu.npu_dump_data import NpuDumpData, DynamicInput
from components.debug.compare.msquickcmp.npu.om_parser import OmParser
from components.debug.compare.msquickcmp.single_op import single_op as sp


def _generate_golden_data_model(args, npu_dump_npy_path):
    if is_saved_model_valid(args.model_path):
        from components.debug.compare.msquickcmp.tf.tf_save_model_dump_data import TfSaveModelDumpData

        return TfSaveModelDumpData(args)
    model_name, extension = utils.get_model_name_and_extension(args.model_path)
    if args.weight_path and ".prototxt" == extension:
        from components.debug.compare.msquickcmp.caffe_model.caffe_dump_data import CaffeDumpData

        return CaffeDumpData(args)
    elif ".pb" == extension:
        from components.debug.compare.msquickcmp.tf.tf_dump_data import TfDumpData

        return TfDumpData(args)
    elif ".onnx" == extension:
        from components.debug.compare.msquickcmp.onnx_model.onnx_dump_data import OnnxDumpData

        return OnnxDumpData(args, npu_dump_npy_path)
    elif ".om" == extension:
        return NpuDumpData(arguments=args, is_golden=True)

    else:
        utils.logger.error("Only model files whose names end with .pb or .onnx or .prototxt are supported")
        raise AccuracyCompareException(utils.ACCURACY_COMPARISON_MODEL_TYPE_ERROR)


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
    try:
        check_and_dump(args, use_cli)
    except utils.AccuracyCompareException as error:
        raise error


def dump_data(args: DumpArgsAdapter, input_shape, original_out_path, use_cli: bool):
    if input_shape:
        args.input_shape = input_shape
        args.out_path = os.path.join(original_out_path, get_shape_to_directory_name(args.input_shape))

    if is_saved_model_valid(args.offline_model_path):
        # npu dump
        from components.debug.compare.msquickcmp.npu.npu_tf_adapter_dump_data import NpuTfAdapterDumpData
        npu_dump = NpuTfAdapterDumpData(args)
        npu_dump.generate_inputs_data()
        npu_dump_data_path, output_json_path = npu_dump.generate_dump_data()
        # gpu dump
        from components.debug.compare.msquickcmp.tf.tf_save_model_dump_data import TfSaveModelDumpData
        golden_dump = TfSaveModelDumpData(args)
        golden_dump.generate_inputs_data(False, npu_dump_data_path, om_parser=None)
        golden_dump.generate_dump_data(output_json_path, npu_dump_path=None, om_parser=None)
    else:
        run_om_model_dump(args, use_cli)


def run_om_model_dump(args, use_cli):
    # whether use aipp
    output_json_path = atc_utils.convert_model_to_json(args.cann_path, args.offline_model_path, args.out_path)
    temp_om_parser = OmParser(output_json_path)
    use_aipp = True if temp_om_parser.get_aipp_config_content() else False

    if use_aipp and args.fusion_switch_file is not None:
        utils.logger.error("if .om model is using aipp config, --fusion-switch-file arg is not support.")
        raise AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_PARAM_ERROR)

    npu_dump = NpuDumpData(args, is_golden=False)
    # generate npu inputs data
    npu_dump.generate_inputs_data(use_aipp=use_aipp)
    # generate npu dump data
    npu_dump_data_path, npu_net_output_data_path = npu_dump.generate_dump_data(use_cli=use_cli)
    # convert data from bin to npy if --convert is used, or if custom_op is not empty
    if args.bin2npy or args.custom_op != "":
        npu_dump_npy_path = convert_bin_dump_data_to_npy(npu_dump_data_path, npu_net_output_data_path, args.cann_path)
    else:
        npu_dump_npy_path = ""
    # generate onnx inputs data
    golden_dump = _generate_golden_data_model(args, npu_dump_npy_path)
    # generate dump data by golden model
    if is_saved_model_valid(args.model_path):
        golden_dump.generate_inputs_data(True, npu_dump_data_path, use_aipp)
        golden_dump.generate_dump_data(output_json_path, npu_dump_npy_path, npu_dump.om_parser)
    else:
        golden_dump.generate_inputs_data(npu_dump_data_path, use_aipp)
        golden_dump.generate_dump_data(npu_dump_npy_path, npu_dump.om_parser)


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

