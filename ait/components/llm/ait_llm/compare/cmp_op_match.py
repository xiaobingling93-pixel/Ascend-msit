# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
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

from enum import Enum
from typing import Any
from ait_llm.common.log import logger
from ait_llm.compare.cmp_data_parse import CompareDataParse
from ait_llm.compare.atb_acc_cmp import load_mapping, pair_built_in_op, pair_custom_op, load_mapping
from ait_llm.dump.torch_dump.topo import ModelTree, TreeNode


class MatchScore(Enum):
    FULL_MATCH = 100
    MAY_MATCH = 50
    NO_MATCH_INFO = 0
    MAY_NOT_MATCH = -50
    NOT_MATCH = -100


class MatchLocation(Enum):
    ALL_INPUT = '/all_input/'
    ALL_OUTPUT = '/all_output/'


class OpMatchMap:

    def __init__(self, golden_data, my_data) -> None:
        self.golden_data = golden_data
        self.my_data = my_data
        self.map: dict = dict()

    def add_score(self, my_op: TreeNode, my_op_location, golden_op: TreeNode, golden_op_location, score: MatchScore):
        '''
        my_op, golden_op: 算子名
        my_op_location, golden_opp_location: 算子输入输出名，就是文件名
        '''
        key = (my_op, my_op_location, golden_op, golden_op_location)
        if key not in self.map:
            self.map[key] = score
        else:
            oriValue = self.map.get(key)
            self.map[key] = oriValue + score

    def get_match_map(self, enable_print: bool = True) -> tuple:
        '''
        返回一个自己数据的算子列表与标杆数据的算子匹配的字典
        '''
        if enable_print:
            pass

        return (
            map_info
            for map_info in self.map.keys()
            if self.map.get(map_info, MatchScore.NO_MATCH_INFO) >= MatchScore.FULL_MATCH
        )


def policy_output(golden_root_node: TreeNode, my_root_node: TreeNode, match_map: OpMatchMap):
    match_map.add_score(
        golden_root_node.children[-1], MatchLocation.ALL_OUTPUT, my_root_node.children[-1], MatchLocation.ALL_OUTPUT
    )


def policy_name_full_match(golden_root_node: TreeNode, my_root_node: TreeNode, match_map: OpMatchMap):
    golden_name2node = {node.name: node for node in golden_root_node.get_all_nodes()}
    my_name2node = {node.name: node for node in golden_root_node.get_all_nodes()}

    golden_name_set = set(golden_name2node.keys())
    my_name_set = set(my_name2node.keys())

    full_match_names = my_name_set.intersection(golden_name_set)

    for match_name in full_match_names:
        match_map.add_score(
            golden_name2node.get(match_name),
            MatchLocation.ALL_OUTPUT,
            my_name2node.get(match_name),
            MatchLocation.ALL_OUTPUT,
        )

        match_map.add_score(
            golden_name2node.get(match_name),
            MatchLocation.ALL_INPUT,
            my_name2node.get(match_name),
            MatchLocation.ALL_INPUT,
        )


class OpMatchMgr:

    def __init__(self, args) -> None:
        self.op_match_policies = [
            policy_output,
            policy_name_full_match,
            OpMatchPolicyMapCount(args),
        ]

    def match(self, golden_data: CompareDataParse, my_data: CompareDataParse):
        match_map = OpMatchMap(golden_data=golden_data, my_data=my_data)
        golden_trees = golden_data.getRootNode()
        my_trees = my_data.getRootNode()
        if len(golden_trees) != len(my_trees):
            # 如果数量不相等，就重复一次。为了处理 atb 有encode+decode两个模型。而 torch只有一个模型的场景
            max_len = max(len(golden_data), len(my_trees))
            golden_trees = (golden_trees * max_len)[0, max_len]
            my_trees = (my_trees * max_len)[0, max_len]
        for golden_model_tree, my_model_tree in zip(golden_trees, my_trees):
            for policy in self.op_match_policies:
                policy(golden_model_tree, my_model_tree, match_map)

        return match_map.get_match_map()


class OpMatchPolicyMapCount:
    def __init__(self, args) -> None:
        self.mapping_dic = load_mapping(args.mapping_file)

    def __call__(self, golden_root_node: TreeNode, my_root_node: TreeNode, match_map: OpMatchMap) -> None:
        golden_layer_type = golden_root_node.get_layer_node_type()
        logger.info("golden_layer_type: %s", golden_layer_type)
        golden_layer_nodes = golden_root_node.get_layer_node(golden_layer_type)

        my_layer_type = my_root_node.get_layer_node_type()
        logger.info("my_layer_type: %s", my_layer_type)
        my_layer_nodes = my_root_node.get_layer_node(my_layer_type)

        def add_output_match(golden_node, golden_op_location, my_node, my_op_location):
            match_map.add_score(my_node, my_op_location, golden_node, golden_op_location, MatchScore.FULL_MATCH)

        # Layer 层对比
        for golden_layer, my_layer in zip(golden_layer_nodes, my_layer_nodes):
            add_output_match(golden_layer, my_layer)

        # 原生算子比对
        for golden_layer, my_layer in zip(golden_layer_nodes, my_layer_nodes):
            g_layer_leaf_nodes = golden_layer.get_leaf_nodes()
            m_layer_leaf_nodes = my_layer.get_leaf_nodes()
            pair_built_in_op(
                g_layer_leaf_nodes,
                m_layer_leaf_nodes,
                self.mapping_dic.get("ATB_TORCH_BUILT_IN_OP_OUTPUT_MAPPING"),
                my_root_node,
                add_output_match,
            )

        # 自定义算子比对
        for golden_layer, my_layer in zip(golden_layer_nodes, my_layer_nodes):
            g_layer_all_nodes = golden_layer.get_all_nodes()
            m_layer_all_nodes = my_layer.get_all_nodes()
            pair_custom_op(
                g_layer_all_nodes,
                m_layer_all_nodes,
                self.mapping_dic.get("ATB_TORCH_CUSTOM_OP_OUTPUT_MAPPING"),
                add_output_match,
            )
