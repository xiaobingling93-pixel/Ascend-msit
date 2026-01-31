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
from auto_optimizer.pattern.knowledge_factory import KnowledgeFactory
from auto_optimizer.pattern.pattern import MatchPattern
from auto_optimizer.pattern.pattern import MatchBase
from auto_optimizer.pattern.pattern import Pattern
from auto_optimizer.pattern.matcher import MatchResult
from auto_optimizer.graph_refactor.interface.base_graph import BaseGraph
from auto_optimizer.graph_refactor.interface.base_node import BaseNode
from auto_optimizer.pattern.knowledges.knowledge_base import KnowledgeBase


class ResizeModeOptimize:
    def __init__(self):
        self._mode_from = [b'linear', b'cubic', b'area', b'random']
        self._mode_to = b'nearest'

    @property
    def mode_from(self):
        return self._mode_from

    @property
    def mode_to(self):
        return self._mode_to


class ResizeOpMatch(MatchBase):
    def __init__(self):
        super().__init__()

    def match(self, node: BaseNode, graph: BaseGraph) -> bool:
        if node is None:
            return False
        if not node.op_type == 'Resize':
            return False
        if node['mode'] in ResizeModeOptimize().mode_from:
            return True
        return False


@KnowledgeFactory.register()
class KnowledgeResizeModeToNearest(KnowledgeBase):
    def __init__(self):
        super().__init__()
        self.resize_op_pattern = Pattern() \
            .add_node("resize_operator", ["Resize"], [ResizeOpMatch()]) \
            .set_node_loop('resize_operator', MatchPattern.MATCH_ONCE) \
            .set_loop(MatchPattern.MATCH_ONCE)
        self._register_apply_funcs(self.resize_op_pattern, [self._resize_mode_apply])

    def _resize_mode_apply(self, graph: BaseGraph, match_result: MatchResult) -> bool:
        """ Resize 模型转换应用方法
        :param graph       : 整图
        :param match_result: 子图匹配结果
        :return            : 模式转换是否应用成功
        """
        mode = ResizeModeOptimize()
        for node_dict in match_result.node_dicts:
            for nodes in node_dict.values():
                for node in nodes:
                    if node.op_type != 'Resize':
                        continue
                    if node['mode'] not in mode.mode_from:
                        continue
                    _node = graph[node.name]
                    _node['mode'] = mode.mode_to
                    if mode.mode_to == b'nearest':
                        _node['nearest_mode'] = b'round_prefer_floor'
        return True
