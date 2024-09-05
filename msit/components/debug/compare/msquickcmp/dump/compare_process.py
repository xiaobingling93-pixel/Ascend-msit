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
This class mainly compare cpu and npu ops inputs and outputs.
"""

import os
import csv
import shutil
import stat
import subprocess
import onnxruntime
import time
import acl

from auto_optimizer.graph_refactor import Node
from components.debug.compare.msquickcmp.net_compare.net_compare import NetCompare
from components.debug.compare.msquickcmp.net_compare import analyser
from components.debug.compare.msquickcmp.common import utils
from components.debug.compare.msquickcmp.adapter_cli.args_adapter import CmpArgsAdapter
from components.debug.compare.msquickcmp.common.utils import get_shape_to_directory_name
from auto_optimizer import OnnxGraph
from components.debug.compare.msquickcmp.accuracy_locat import accuracy_locat as al
from components.debug.compare.msquickcmp.common.utils import AccuracyCompareException

ERROR_INTERVAL_INFO_FILE = "error_interval_info.txt"
WRITE_MODES = stat.S_IWUSR | stat.S_IRUSR
READ_WRITE_FLAGS = os.O_RDWR | os.O_CREAT


def compare_process(args: CmpArgsAdapter):
    # compare the entire network
    if args.my_path is None or args.golden_path is None or args.ops_json is None:
        raise ValueError("when compare alone. Please ensure that both --my-path and --golden-path and --ops-json are "
                         "provided.")
    net_compare = NetCompare(args.my_path, args.golden_path,
                             args.ops_json, args, golden_json_path=None)
    net_compare.accuracy_network_compare()

    if not args.locat:
        invalid_rows, _ = analyser.Analyser(args.out_path)()
    else:
        invalid_rows, _ = analyser.Analyser(args.out_path)('ALL_INVALID')
    print_advisor_info(args.out_path)
    _append_is_npu_ops_to_csv(args.out_path)

    return invalid_rows


def compare_run(args: CmpArgsAdapter):
    res = compare_process(args)
    if res and args.locat:
        endnode_names_list = res[0]["GroundTruth"].split(",")
        endnode_name = endnode_names_list[0]
        error_node_list = find_accuracy_interval(args, endnode_name, input_shape="")
        error_interval_info_file = os.path.join(args.out_path, ERROR_INTERVAL_INFO_FILE)
        with os.fdopen(os.open(error_interval_info_file, READ_WRITE_FLAGS, WRITE_MODES), "a+") as fp_writer:
            output_error_interval_info(fp_writer, error_node_list)


def find_accuracy_interval(args, endnode_name, input_shape):
    """
    Function:
        find accuracy interval of the error node
    Return:
        an error node interval list
    """
    if input_shape:
        args.out_path = os.path.join(args.out_path, get_shape_to_directory_name(input_shape))

    # 读入onnx数据文件的路径
    onnx_file_path = 'dump_data/onnx'
    onnx_data_path = os.path.join(args.out_path, onnx_file_path)

    # 读入onnx模型
    og = OnnxGraph.parse(args.model_path)
    og.infer_shape()

    # 获取精度异常节点
    endnode = og.get_node(endnode_name, node_type=Node)

    output_file = './accuracy_location_log.txt'
    output_file = os.path.realpath(output_file)
    error_node_list = []
    # 验证单层算子是否有问题
    node_interval = [endnode, endnode]
    # 单层算子无问题
    if not subgraph_check(og, node_interval, args, onnx_data_path, input_shape):
        for node in og.nodes:
            if al.check_input_node(og, node):
                input_node_interval = [node, endnode]
                l_node, r_node = bin_divide(og, input_node_interval, args, onnx_data_path, input_shape)
                utils.logger.info("Accumulated Error interval has been found.")
                error_node_list.append([l_node, r_node])
        return error_node_list
    return [[endnode, endnode]]


def subgraph_check(og, node_interval, args, onnx_data_path, input_shape):
    startnode, endnode = node_interval
    subgraph_onnx_file = os.path.join(args.out_path, 'tmp_for_accuracy_locat.onnx')
    try:
        og.extract_subgraph([startnode.name], [endnode.name], subgraph_onnx_file)
    except Exception as e:
        utils.logger.error("Failed to extract subgraph model")
        raise AccuracyCompareException(utils.ACCRACY_COMPARISON_EXTRACT_ERROR) from e
    utils.logger.info("Extracting model Sucess!")
    utils.logger.info("Start using atc to convert onnx to om file")
    subgraph_om_file = os.path.join(args.out_path, 'tmp_for_accuracy_locat')
    atc_cmd = ["atc", "--framework=5", "--soc_version=" + acl.get_soc_name(), "--model=" + subgraph_onnx_file, \
               "--output=" + subgraph_om_file]
    subprocess.run(atc_cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    utils.logger.info("atc conversion Success!")
    utils.logger.info("Start to loading input data")
    OnnxGraph.parse(subgraph_onnx_file)
    inputs_list = [(ii.name, ii.shape) for ii in onnxruntime.InferenceSession(subgraph_onnx_file).get_inputs()]
    input_need_list = al.input_completion(og, inputs_list)
    pattern = '|'.join(input_need_list)
    try:
        matched_files = al.find_npy_files_with_prefix(onnx_data_path, pattern)
    except Exception as e:
        utils.logger.error("Failed to find onnx dump data, please check whether file path is right")
        raise AccuracyCompareException(utils.ACCRACY_COMPARISON_FETCH_DATA_ERROR) from e
    sort_matched_files = []
    for prefix in input_need_list:
        for match_file in matched_files:
            file_name = os.path.basename(match_file)
            if file_name.startswith(prefix):
                sort_matched_files.append(match_file)
    bin_files_path = al.create_bin_file(args.out_path, sort_matched_files)
    tmp_bin_path = os.path.join(args.out_path, 'tmp')
    utils.logger.info("Loading data Finished!")
    tmp_out_path = os.path.join(args.out_path, 'tmpres')
    if not os.path.exists(tmp_out_path):
        os.makedirs(tmp_out_path)
    time_dir = time.strftime("%Y%m%d%H%M%S", time.localtime())
    original_out_path = os.path.realpath(os.path.join(args.out_path, time_dir))
    cmg_args = CmpArgsAdapter(subgraph_onnx_file, os.path.join(args.out_path, "tmp_for_accuracy_locat.om"),
                              "", bin_files_path, args.cann_path, tmp_out_path, "", args.device,
                              "", "", False, "", True, False, custom_op=args.custom_op, locat=True)
    utils.logger.info("Start to run comparision")
    res = run(cmg_args, input_shape, original_out_path, True)
    utils.logger.info("Comparision finished")
    shutil.rmtree(tmp_out_path)
    shutil.rmtree(tmp_bin_path)
    if al.check_res(res, endnode):
        return True
    return False


def bin_divide(og, node_interval, args, onnx_data_path, input_shape):
    """
    Function:
        using binary search to find the accuracy error interval
    Return:
        an accuracy error interval list
    """
    startnode, endnode = node_interval
    subgraph_model_path = os.path.join(args.out_path, 'tmp_for_subgraph.onnx')
    og.extract_subgraph([startnode.name], [endnode.name], subgraph_model_path)
    subog = OnnxGraph.parse(subgraph_model_path)

    utils.logger.info("Binary Search for error interval starts.")
    # 直线化
    satisfied_nodes = []
    satisfied_nodes = al.calculate_flow(subog, startnode, endnode)
    low = 0
    high = len(satisfied_nodes) - 1

    # 二分
    while low < high:
        mid = (low + high + 1) // 2
        input_node_interval = [satisfied_nodes[mid], endnode]
        if subgraph_check(og, input_node_interval, args, onnx_data_path, input_shape):
            low = mid
        else:
            high = mid - 1
    utils.logger.info("Binary Search for error interval ends.")
    return satisfied_nodes[low], endnode


def output_error_interval_info(fp_writer, error_node_list):
    for [l_node, r_node] in error_node_list:
        fp_writer.write(f"{l_node}:{r_node}")


def print_advisor_info(out_path):
    advisor_info_txt_path = os.path.join(out_path, 'advisor_summary.txt')
    if os.path.exists(advisor_info_txt_path):
        utils.logger.info(f"The advisor summary (.txt) is saved in :\"{advisor_info_txt_path}\"")
        with open(advisor_info_txt_path, 'r') as advisor_file:
            lines = advisor_file.readlines()
            for line in lines:
                utils.logger.info(line.strip())


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


def _get_single_csv_in_folder(csv_path):
    for file_name in os.listdir(csv_path):
        if file_name.endswith('.csv'):
            return os.path.join(csv_path, file_name)
    raise IOError(f"None csv file exists in folder {csv_path}")


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


def _correct_the_wrong_order(left_index, right_index, golden_net_output_info):
    if left_index not in golden_net_output_info.keys() or right_index not in golden_net_output_info.keys():
        return
    if left_index != right_index:
        tmp = golden_net_output_info[left_index]
        golden_net_output_info[left_index] = golden_net_output_info[right_index]
        golden_net_output_info[right_index] = tmp
        utils.logger.info('swap the %s and %s item in golden_net_output_info!', left_index, right_index)
