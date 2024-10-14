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

import itertools
import re
from enum import Enum
from msit_llm.common.log import logger
from msit_llm.dump.torch_dump.topo import TreeNode
from msit_llm.compare.op_mapping import ATB_QUANT_FLOAT_NODE_MAPPING


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
            self.map[key] = score.value
        else:
            ori_value = self.map.get(key)
            self.map[key] = ori_value + score.value

    def get_match_map(self, enable_print: bool = True) -> tuple:
        '''
        返回一个自己数据的算子列表与标杆数据的算子匹配的字典
        '''
        if enable_print:
            max_length = 0
            for map_info in self.map.keys():
                my_op, my_op_location, golden_op, golden_op_location = map_info
                max_length = max(len(f"[{my_op.node_name}#{my_op.op_type}#{my_op_location}]"), max_length)

            format_str = f"mapping score: %-{max_length}s   <- %s ->   %s "
            for map_info, score in self.map.items():
                my_op, my_op_location, golden_op, golden_op_location = map_info
                logger.debug(
                    format_str,
                    f"[{my_op.node_name}#{my_op.op_type}#{my_op_location}]",
                    str(score),
                    f"[{golden_op.node_name}#{golden_op.op_type}#{golden_op_location}]",
                )

        ret_map = [
            map_info
            for map_info in self.map.keys()
            if self.map.get(map_info, MatchScore.NO_MATCH_INFO) >= MatchScore.FULL_MATCH.value
        ]

        ret_map.sort(key=lambda item: item[0].show_order)  # 根据my_op 的 show_order 进行排序
        return ret_map


def policy_output(golden_root_node: TreeNode, my_root_node: TreeNode, match_map: OpMatchMap):
    # 最后输出匹配上
    def get_last_child(node):
        if hasattr(node, "children") and node.children is not None and len(node.children) > 0:
            return node.children[-1]
        else:
            return None

    def get_all_last_child(node):
        all_last_nodes = []
        next_last_node = node
        while next_last_node is not None:
            next_last_node = get_last_child(next_last_node)
            if next_last_node is None:
                break
            all_last_nodes.append(next_last_node)

        return all_last_nodes

    my_last_nodes = get_all_last_child(my_root_node)
    golden_last_nodes = get_all_last_child(golden_root_node)

    for golden_node, my_node in itertools.product(golden_last_nodes, my_last_nodes):
        match_map.add_score(
            my_node,
            MatchLocation.ALL_OUTPUT,
            golden_node,
            MatchLocation.ALL_OUTPUT,
            MatchScore.FULL_MATCH,
        )


def policy_name_full_match(golden_root_node: TreeNode, my_root_node: TreeNode, match_map: OpMatchMap):
    # 名字一样就代表匹配上了
    golden_name2node = {node.node_name: node for node in golden_root_node.get_all_nodes()}
    my_name2node = {node.node_name: node for node in my_root_node.get_all_nodes()}

    golden_name_set = set(golden_name2node.keys())
    my_name_set = set(my_name2node.keys())

    full_match_names = my_name_set.intersection(golden_name_set)

    for match_name in full_match_names:
        if match_name == "root":
            continue
        match_map.add_score(
            my_name2node.get(match_name),
            MatchLocation.ALL_OUTPUT,
            golden_name2node.get(match_name),
            MatchLocation.ALL_OUTPUT,
            MatchScore.FULL_MATCH,
        )


op_mapping_dict = {
    # ATB算子: Pytorch算子
    ('qkv', 'linearoperation'): 'self_attn.w_pack',
    ('attention', 'linearoperation'): 'self_attn.o_proj',
    ('mlp', 'rmsnormoperation'): 'mlp.post_attention_layernorm',
    ('mlp', 'splitoperation'): 'mlp.gate_proj',
    ('mlp', 'activationoperation'): 'mlp.act_fn',
    ('mlp', 'elewiseoperation'): 'mlp.down_proj',
    ('mlp', 'linearoperation'): 'mlp.down_proj'
}

golden_op_data_mapping = {
    # 算子名：文件名 (因为pytorch算子mlp.down_proj匹配了两次无法区分，使用匹配到的ATB算子做键值)
    ('mlp', 'elewiseoperation'): 'input_0.pth'
}

my_op_data_mapping = {
    # 算子名：文件名
    ('mlp', 'splitoperation'): 'outtensor0.bin'
}


def policy_enhanced_name_match(golden_root_node: TreeNode, my_root_node: TreeNode, match_map: OpMatchMap):
    def match_ops(golden_path_dict, my_path_dict, golden_path2node, my_path2node, match_map):
        for golden_path_lower, golden_path in golden_path_dict.items():
            for my_path_lower, my_path in my_path_dict.items():
                for atb_op, torch_op in op_mapping_dict.items():
                    atb_op_lower = atb_op.lower() if isinstance(atb_op, str) else [op.lower() for op in atb_op]
                    torch_op_lower = torch_op.lower()

                    if all(op in my_path_lower for op in atb_op_lower) and torch_op_lower in golden_path_lower:
                        if torch_op_lower == 'self_attn.o_proj' and 'qkv' in my_path_lower:
                            continue

                        g_match_location = golden_op_data_mapping.get(tuple(atb_op_lower), MatchLocation.ALL_OUTPUT)
                        m_match_location = my_op_data_mapping.get(tuple(atb_op_lower), MatchLocation.ALL_OUTPUT)

                        match_map.add_score(
                            my_path2node.get(my_path),
                            m_match_location,
                            golden_path2node.get(golden_path),
                            g_match_location,
                            MatchScore.FULL_MATCH,
                        )

    golden_path2node = {node.tensor_path: node for node in golden_root_node.get_all_nodes()}
    my_path2node = {node.tensor_path: node for node in my_root_node.get_all_nodes()}

    golden_layer_type = golden_root_node.get_layer_node_type()
    golden_layer_nodes = golden_root_node.get_layer_node(golden_layer_type)

    my_layer_type = my_root_node.get_layer_node_type()
    my_layer_nodes = my_root_node.get_layer_node(my_layer_type)

    for golden_layer, my_layer in zip(golden_layer_nodes, my_layer_nodes):
        golden_path_dict = {node.tensor_path.lower(): node.tensor_path for node in golden_layer.get_leaf_nodes()}
        my_path_dict = {node.tensor_path.lower(): node.tensor_path for node in my_layer.get_leaf_nodes()}
        match_ops(golden_path_dict, my_path_dict, golden_path2node, my_path2node, match_map)


outside_layer_op_mapping_dict = {
    # ATB算子: Pytorch算子
    'wordembed': 'embed_tokens',
    re.compile(r"^RmsNormOperation_\d+$"): 'root.model.norm',
    'lmhead': 'lm_head'
}


def policy_outside_layer_name_match(golden_root_node: TreeNode, my_root_node: TreeNode, match_map: OpMatchMap):
    golden_name2node = {node.node_name: node for node in golden_root_node.get_all_nodes()}
    my_name2node = {node.node_name: node for node in my_root_node.get_all_nodes()}

    golden_name_set = set(golden_name2node.keys())
    my_name_set = set(my_name2node.keys())

    def match_names(name_set, pattern):
        if isinstance(pattern, re.Pattern):
            return [name for name in name_set if pattern.match(name)]
        else:
            return [name for name in name_set if pattern in name.lower()]

    for atb_op, torch_op in outside_layer_op_mapping_dict.items():
        my_matches = match_names(my_name_set, atb_op)
        golden_matches = match_names(golden_name_set, torch_op)

        if golden_matches and my_matches:
            for golden_name, my_name in zip(golden_matches, my_matches):
                match_map.add_score(
                    my_name2node.get(my_name),
                    MatchLocation.ALL_OUTPUT,
                    golden_name2node.get(golden_name),
                    MatchLocation.ALL_OUTPUT,
                    MatchScore.FULL_MATCH,
                )


def policy_layer_type_cnt_match(golden_root_node: TreeNode, my_root_node: TreeNode, match_map: OpMatchMap):
    # 逐层比对，类型和数量一致的算匹配上。主要用于量化和非量化的atb 之间比对
    def get_children_type_count_map(node):
        # 汇总 children 的每个类型的 node 到一个 map 中
        type_cnt_map = {}
        if node is None or node.children is None or len(node.children) == 0:
            return type_cnt_map
        for child in node.children:
            save_in_type = ATB_QUANT_FLOAT_NODE_MAPPING.get(child.op_type, child.op_type)
            type_cnt_map.setdefault(save_in_type, []).append(child)
        return type_cnt_map

    matched_node_map = [(golden_root_node, my_root_node)]
    comparing_index = 0
    while comparing_index < len(matched_node_map):
        golden_node, my_node = matched_node_map[comparing_index]
        comparing_index = comparing_index + 1

        golden_type_count_map = get_children_type_count_map(golden_node)
        my_type_count_map = get_children_type_count_map(my_node)

        for op_type, my_nodes in my_type_count_map.items():
            if len(my_nodes) != len(golden_type_count_map.get(op_type, [])):
                continue
            golden_nodes = golden_type_count_map.get(op_type)
            matched_node_map.extend(zip(golden_nodes, my_nodes))

    for golden_node, my_node in matched_node_map:
        match_map.add_score(
            my_node,
            MatchLocation.ALL_OUTPUT,
            golden_node,
            MatchLocation.ALL_OUTPUT,
            MatchScore.FULL_MATCH,
        )


def policy_rope_operator_match(golden_root_node: TreeNode, my_root_node: TreeNode, match_map: OpMatchMap):
    golden_name2node = {node.node_name: node for node in golden_root_node.get_all_nodes()}
    my_name2node = {node.node_name: node for node in my_root_node.get_all_nodes()}

    golden_layer_type = golden_root_node.get_layer_node_type()
    golden_layer_nodes = golden_root_node.get_layer_node(golden_layer_type)

    my_layer_type = my_root_node.get_layer_node_type()
    my_layer_nodes = my_root_node.get_layer_node(my_layer_type)

    for golden_layer, my_layer in zip(golden_layer_nodes, my_layer_nodes):
        g_layer_leaf_nodes = golden_layer.get_leaf_nodes()
        m_layer_leaf_nodes = my_layer.get_leaf_nodes()

        golden_name_set = [node.node_name for node in g_layer_leaf_nodes]
        my_name_set = [node.node_name for node in m_layer_leaf_nodes]

        golden_rotary_name_set = [item for item in golden_name_set if "rotary" in item]
        my_rope_name_set = [item for item in my_name_set if "ropeoperation" in item.lower()]

        for golden_node, my_node in zip(golden_rotary_name_set, my_rope_name_set):
            match_map.add_score(
                my_name2node.get(my_node),
                MatchLocation.ALL_INPUT,
                golden_name2node.get(golden_node),
                MatchLocation.ALL_OUTPUT,
                MatchScore.FULL_MATCH,
            )


class OpMatchMgr:

    def __init__(self, args) -> None:
        self.op_match_policies = [
            policy_output,
            policy_name_full_match,
            policy_layer_type_cnt_match,
            OpMatchPolicyMapCount(args),
            policy_rope_operator_match,
            policy_enhanced_name_match,
            policy_outside_layer_name_match,
        ]

    def match(self, golden_data, my_data):
        match_map = OpMatchMap(golden_data=golden_data, my_data=my_data)
        golden_trees = golden_data.get_root_nodes()
        my_trees = my_data.get_root_nodes()
        if len(golden_trees) != len(my_trees):
            # 如果数量不相等，就重复一次。为了处理 atb 有encode+decode两个模型。而 torch只有一个模型的场景
            max_len = max(len(golden_trees), len(my_trees))
            golden_trees = (golden_trees * max_len)[0:max_len]
            my_trees = (my_trees * max_len)[0:max_len]
        for golden_model_tree, my_model_tree in zip(golden_trees, my_trees):
            for policy in self.op_match_policies:
                policy(golden_model_tree, my_model_tree, match_map)

        return match_map.get_match_map()


class OpMatchPolicyMapCount:
    # 原始 atb 和 torch 的比对逻辑。先找到 block，再取 block 内部类型和数量一致的算子作为匹配
    def __init__(self, args) -> None:
        from msit_llm.compare.atb_acc_cmp import load_mapping

        self.mapping_dic = load_mapping(args.mapping_file)

    def __call__(self, golden_root_node: TreeNode, my_root_node: TreeNode, match_map: OpMatchMap) -> None:
        from msit_llm.compare.atb_acc_cmp import pair_built_in_op, pair_custom_op

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
            add_output_match(golden_layer, MatchLocation.ALL_OUTPUT, my_layer, MatchLocation.ALL_OUTPUT)

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
