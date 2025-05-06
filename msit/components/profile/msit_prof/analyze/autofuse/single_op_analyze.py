# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd. All rights reserved.
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
import subprocess
from collections import defaultdict

import pandas as pd
import numpy as np

from components.utils.util import filter_cmd
from components.utils.log import logger
from msit_prof.analyze.parser.ge_graph import get_all_subgraph


AUTO_FUSE_OP_TYPE = ["AscBackend", "FusedAscBackend"]
PERFORMANCE_COLUMNS = ["Task Duration(us)", "Op Name", "OP Type"]
NO_PROF_OP_TYPE_LIST = ["Reshape", "ExpandDims"]


class OpInfo:
    def __init__(self, op_name, op_type="unknown"):
        self.op_name = op_name
        self.op_type = op_type
        self.duration_time = None
    
    def __str__(self):
        return "(%s, %s)" % (self.op_name, self.op_type)


class SingleOpAnalyzer:
    def __init__(self, args):
        self.fused_op_summary = args.fused
        self.origin_op_summary = args.origin
        self.output_path = args.output
        self.ge_graph_path = args.ops_graph
        self.ops_mapping_json = os.path.join(self.output_path, "ge_proto_build.json")
        self.fuse_df = None
        self.origin_df = None
        self.ops_mapping_dict = None
        self.fused_name_to_type = dict()

    @staticmethod
    def compute_performance_diff(single_fused_df, origin_op_df, analyze_result, fused_name):
        if single_fused_df.empty:
            logger.error(f"{fused_name} can not found fused profiling data.")
            raise ValueError("The file specified by the '--fused' parameter does not match the GE dump graph.")
        single_fused_df = single_fused_df[PERFORMANCE_COLUMNS]
        fused_duration_sum = sum(single_fused_df["Task Duration(us)"].to_list())
        analyze_result["Fused Durations(us)"].append(fused_duration_sum)
        if origin_op_df.empty:
            return
        origin_op_df = origin_op_df[PERFORMANCE_COLUMNS]
        origin_duration_sum = sum(origin_op_df["Task Duration(us)"].to_list())
        analyze_result["Origin Durations(us)"].append(origin_duration_sum)
        analyze_result["Time Ratio"].append(fused_duration_sum / origin_duration_sum)
        analyze_result["Time Difference"].append(fused_duration_sum - origin_duration_sum)

    def convert_ge_graph(self):
        atc_cmd = [
            "atc", "--mode=5", "--om=" + self.ge_graph_path, "--json=" + self.ops_mapping_json
        ]
        atc_cmd = filter_cmd(atc_cmd)
        res = subprocess.run(atc_cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if res.returncode == 0:
            logger.info("atc conversion success!") 
        else:
            raise RuntimeError("atc run failed! Make sure that you have correctly activate CANN environments.")

    def get_fuse_graph_to_origin_op_mapping(self):
        fuse_graph_to_origin_op = defaultdict(list)
        for graph in get_all_subgraph(self.ops_mapping_json):
            for op in graph['op']:
                if op['type'] not in ["AscBackend", "FusedAscBackend"]:
                    continue
                origin_op_list = None
                origin_op_type_list = None
                fuse_op_name = op['name']   
                for attr in op["attr"]:
                    if attr['key'] == "_datadump_original_op_names":
                        origin_op_list = attr['value']["list"]['s']
                    if attr['key'] == "_datadump_original_op_types":
                        origin_op_type_list = attr['value']["list"]['s']
                if origin_op_list and origin_op_type_list:
                    for name, op_type in zip(origin_op_list, origin_op_type_list):
                        fuse_graph_to_origin_op[fuse_op_name].append(OpInfo(name, op_type))
                else:
                    err_msg = f"{fuse_op_name} must both have '_datadump_original_op_names'" + \
                              " and '_datadump_original_op_types' attribute!"
                    raise ValueError(err_msg)
        self.ops_mapping_dict = fuse_graph_to_origin_op

    def load_op_summary(self):
        self.fuse_df = pd.read_csv(self.fused_op_summary, dtype={"Op Name": str})
        self.origin_df = pd.read_csv(self.origin_op_summary, dtype={"Op Name": str})
        filtered_df = self.fuse_df[self.fuse_df["OP Type"].isin(["AscBackend", "FusedAscBackend"])]
        can_compare_fuse_nodes = set() # 可以比对性能增益的融合节点
        for _, row in filtered_df.iterrows():
            fuse_name = row["Op Name"]
            can_compare_fuse_nodes.add(fuse_name)
            if fuse_name not in self.fused_name_to_type:
                self.fused_name_to_type[fuse_name] = row["OP Type"]
        logger.info(f"There are {len(can_compare_fuse_nodes)} AscBc nodes.")
        return can_compare_fuse_nodes

    def get_filter_origin_df(self, fuse_node_name, analyze_result):
        if fuse_node_name not in self.ops_mapping_dict:
            logger.warning(f"{fuse_node_name} not found in GE dump graph, maybe it's info is in another graph.")
            return None
        origin_op_list = self.ops_mapping_dict[fuse_node_name]
        origin_op_name_list = [i.op_name for i in origin_op_list]
        analyze_result["Origin Ops"].append("; ".join([str(i) for i in origin_op_list]))
        origin_op_df = self.origin_df[self.origin_df["Op Name"].isin(origin_op_name_list)]
        if origin_op_df.empty:
            logger.warning(f"{fuse_node_name} can not found origin profiling data based on ge graph.")
        return origin_op_df
    
    def analyze_origin_ops(self, fuse_node_name, total_origin_op_name):
        not_found_op_list = []
        origin_op_duration_sum = []
        for op in self.ops_mapping_dict[fuse_node_name]:
            origin_single_df = self.origin_df[self.origin_df["Op Name"] == op.op_name]
            duration_time = str(origin_single_df["Task Duration(us)"].sum())
            origin_op_duration_sum.append(f"({op.op_name}, {duration_time})")
            # 有些算子不会上device计算，因此采集不到profiling数据，要进行过滤
            if op.op_name not in total_origin_op_name and op.op_type not in NO_PROF_OP_TYPE_LIST:
                not_found_op_list.append(op.op_name)
        return origin_op_duration_sum, not_found_op_list

    def save_analyze_result(self, analyze_result):
        result_df = pd.DataFrame(analyze_result)
        analyze_result_path = os.path.join(self.output_path, "profile_analyse.csv")
        sort_df = result_df.sort_values(by="Time Difference", ascending=True, na_position='last')
        sort_df.to_csv(analyze_result_path, index=False, na_rep='NaN')
        logger.info("analyze result save to %r" % analyze_result_path)

    def analyze(self):
        # 将dump ge_build.txt转换成json文件，方便读取
        self.convert_ge_graph()
        self.get_fuse_graph_to_origin_op_mapping()
        analyze_result = defaultdict(list)
        can_compare_fuse_nodes = self.load_op_summary()
        total_origin_op_name = set(self.origin_df["Op Name"]) # profiling采集到的算子
        for fuse_node_name in can_compare_fuse_nodes:
            origin_op_duration_sum, not_found_op_list = self.analyze_origin_ops(fuse_node_name, total_origin_op_name)
            single_fused_df = self.fuse_df[self.fuse_df["Op Name"] == fuse_node_name]
            analyze_result["Fuse OpName"].append(fuse_node_name)
            analyze_result["Fuse OpType"].append(self.fused_name_to_type[fuse_node_name])
            origin_op_df = self.get_filter_origin_df(fuse_node_name, analyze_result)
            self.compute_performance_diff(single_fused_df, origin_op_df, analyze_result, fuse_node_name)
            if origin_op_df.empty:
                analyze_result["Origin Durations(us)"].append(np.nan)
                analyze_result["Time Ratio"].append(np.nan)
                analyze_result["Time Difference"].append(np.nan)
            if not_found_op_list:
                analyze_result["Time Ratio"][-1] = np.nan
                analyze_result["Time Difference"][-1] = np.nan
            analyze_result["Origin Duration(us) Each Op"].append("; ".join(origin_op_duration_sum))
            analyze_result["Not Found Origin Op"].append("; ".join(not_found_op_list))
        self.save_analyze_result(analyze_result)