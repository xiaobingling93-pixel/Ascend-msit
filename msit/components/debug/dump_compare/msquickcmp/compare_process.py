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

import os
import csv

from components.debug.compare.msquickcmp.net_compare.net_compare import NetCompare
from components.debug.compare.msquickcmp.net_compare import analyser
from components.debug.compare.msquickcmp.common import utils
from msquickcmp.adapter_cli.args_adapter import CompareArgsAdapter


def compare_process(args: CompareArgsAdapter):
    if not args.dump:
        # only compare the final output
        net_compare = NetCompare(args.my_net_output_path, args.golden_path,
                                 args.ops_json, args, golden_json_path=None)
        net_compare.net_output_compare(args.my_net_output_path, args.golden_net_output_path)
    else:
        # compare the entire network
        net_compare = NetCompare(args.my_path, args.golden_path,
                                 args.ops_json, args, golden_json_path=None)
        net_compare.accuracy_network_compare()

    # Check and correct the mapping of net output node name.
    if len(args.expect_net_output_node) == 1:
        _check_output_node_name_mapping(args.expect_net_output_node, args.golden_net_output_path)
    if not args.locat:
        invalid_rows, _ = analyser.Analyser(args.out_path)()
    else:
        invalid_rows, _ = analyser.Analyser(args.out_path)('ALL_INVALID')
    print_advisor_info(args.out_path)
    _append_is_npu_ops_to_csv(args.out_path)


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
