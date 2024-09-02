# coding=utf-8
# Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
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

import csv
import os
import stat
import time

import acl
from msquickcmp.adapter_cli.args_adapter import CmpArgsAdapter
from msquickcmp.atc import atc_utils
from msquickcmp.common import utils
from msquickcmp.common.args_check import is_saved_model_valid
from msquickcmp.common.convert import convert_bin_dump_data_to_npy
from msquickcmp.common.convert import convert_npy_to_bin
from msquickcmp.common.utils import AccuracyCompareException, get_shape_to_directory_name
from msquickcmp.npu.npu_dump_data import NpuDumpData, DynamicInput
from msquickcmp.npu.om_parser import OmParser
from msquickcmp.single_op import single_op as sp

WRITE_MODES = stat.S_IWUSR | stat.S_IRUSR
READ_WRITE_FLAGS = os.O_RDWR | os.O_CREAT
WRITE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
ERROR_INTERVAL_INFO_FILE = "error_interval_info.txt"
MAX_MEMORY_USE = 6 * 1024 * 1024 * 1024


def _generate_golden_data_model(args, npu_dump_npy_path):
    if is_saved_model_valid(args.model_path):
        from msquickcmp.tf.tf_save_model_dump_data import TfSaveModelDumpData

        return TfSaveModelDumpData(args)
    model_name, extension = utils.get_model_name_and_extension(args.model_path)
    if args.weight_path and ".prototxt" == extension:
        from msquickcmp.caffe_model.caffe_dump_data import CaffeDumpData

        return CaffeDumpData(args)
    elif ".pb" == extension:
        from msquickcmp.tf.tf_dump_data import TfDumpData

        return TfDumpData(args)
    elif ".onnx" == extension:
        from msquickcmp.onnx_model.onnx_dump_data import OnnxDumpData

        return OnnxDumpData(args, npu_dump_npy_path)
    elif ".om" == extension:
        return NpuDumpData(arguments=args, is_golden=True)

    else:
        utils.logger.error("Only model files whose names end with .pb or .onnx or .prototxt are supported")
        raise AccuracyCompareException(utils.ACCURACY_COMPARISON_MODEL_TYPE_ERROR)


def _correct_the_wrong_order(left_index, right_index, golden_net_output_info):
    if left_index not in golden_net_output_info.keys() or right_index not in golden_net_output_info.keys():
        return
    if left_index != right_index:
        tmp = golden_net_output_info[left_index]
        golden_net_output_info[left_index] = golden_net_output_info[right_index]
        golden_net_output_info[right_index] = tmp
        utils.logger.info('swap the %s and %s item in golden_net_output_info!', left_index, right_index)


def _check_output_node_name_mapping(original_net_output_node, golden_net_output_info):
    for left_index, node_name in original_net_output_node.items():
        match = False
        for right_index, dump_file_path in golden_net_output_info.items():
            dump_file_name = os.path.basename(dump_file_path)
            if dump_file_name.startswith(node_name.replace("/", "_").replace(":", ".")):
                match = True
                _correct_the_wrong_order(left_index, right_index, golden_net_output_info)
                break
        if not match:
            utils.logger.warning("the original name: {} of net output maybe not correct!".format(node_name))
            break


def _get_single_csv_in_folder(csv_path):
    for file_name in os.listdir(csv_path):
        if file_name.endswith('.csv'):
            return os.path.join(csv_path, file_name)
    raise IOError(f"None csv file exists in folder {csv_path}")


def _append_is_npu_ops_to_csv(csv_path):
    csv_path = _get_single_csv_in_folder(csv_path)
    if os.path.islink(csv_path):
        os.unlink(csv_path)
    if os.path.exists(csv_path):
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            rows = [row for row in reader]
        header = rows[0]
        ground_truth_col = header.index("GroundTruth")
        header.append('IsNpuOps')
        for row in rows[1:]:
            is_npu_ops = "YES" if row[ground_truth_col] == "*" else "NO"
            row.append(is_npu_ops)
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)


def cmp_process(args: CmpArgsAdapter, use_cli: bool):
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


def dump_data(args: CmpArgsAdapter, input_shape, original_out_path, use_cli: bool):
    if input_shape:
        args.input_shape = input_shape
        args.out_path = os.path.join(original_out_path, get_shape_to_directory_name(args.input_shape))

    if is_saved_model_valid(args.offline_model_path):
        # npu dump
        from msquickcmp.npu.npu_tf_adapter_dump_data import NpuTfAdapterDumpData
        npu_dump = NpuTfAdapterDumpData(args)
        npu_dump.generate_inputs_data()
        npu_dump_data_path, output_json_path = npu_dump.generate_dump_data()
        # gpu dump
        from msquickcmp.tf.tf_save_model_dump_data import TfSaveModelDumpData
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


def print_advisor_info(out_path):
    advisor_info_txt_path = os.path.join(out_path, 'advisor_summary.txt')
    if os.path.exists(advisor_info_txt_path):
        utils.logger.info(f"The advisor summary (.txt) is saved in :\"{advisor_info_txt_path}\"")
        with open(advisor_info_txt_path, 'r') as advisor_file:
            lines = advisor_file.readlines()
            for line in lines:
                utils.logger.info(line.strip())


def fusion_close_model_convert(args: CmpArgsAdapter):
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


def check_and_dump(args: CmpArgsAdapter, use_cli: bool):
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

