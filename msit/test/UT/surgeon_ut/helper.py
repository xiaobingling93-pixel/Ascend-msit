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
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from numpy.typing import NDArray
import onnxruntime as ort

from auto_optimizer.common.utils import meet_precision
from auto_optimizer.graph_optimizer.optimizer import GraphOptimizer
from auto_optimizer.graph_refactor.interface.base_graph import BaseGraph
from auto_optimizer.pattern.knowledges import KnowledgeBigKernel
from auto_optimizer.pattern.knowledges.knowledge_base import KnowledgeBase

ort.set_default_logger_severity(3)


@dataclass
class OptimizationConfig:
    graph: BaseGraph
    knowledge: KnowledgeBase
    onnx_ori: str = ''
    onnx_opt: str = ''


class KnowledgeTestHelper:
    @staticmethod
    def graph_equal(lhs: BaseGraph, rhs: BaseGraph) -> bool:
        '''检查两个图是否等价，检查图的inputs/outputs/initializers/nodes的相对关系是否相同。
        比如如果只是做infershape，则认为前后的图仍是同一个。
        '''
        if not (isinstance(rhs, BaseGraph) and isinstance(lhs, BaseGraph)):
            return False
        if lhs.name != rhs.name:
            return False
        inputs_lhs = {inp.name: inp for inp in lhs.inputs}
        inputs_rhs = {inp.name: inp for inp in rhs.inputs}
        if inputs_lhs != inputs_rhs:
            return False
        outputs_lhs = {out.name: out for out in lhs.outputs}
        outputs_rhs = {out.name: out for out in rhs.outputs}
        if outputs_lhs != outputs_rhs:
            return False
        inits_lhs = {ini.name: ini for ini in lhs.initializers}
        inits_rhs = {ini.name: ini for ini in rhs.initializers}
        if inits_lhs != inits_rhs:
            return False
        inmap_lhs, inmap_rhs = defaultdict(dict), defaultdict(dict)
        for node in lhs.nodes:
            for inp in node.inputs:
                inmap_lhs[inp][node.name] = node
        for node in rhs.nodes:
            for inp in node.inputs:
                inmap_rhs[inp][node.name] = node
        return inmap_lhs == inmap_rhs

    @staticmethod
    def inference(onnx_path: str, feeds: List[Dict[str, NDArray]]) -> List[List[NDArray]]:
        '''Inference a onnx model with a list of feeds'''
        session = ort.InferenceSession(onnx_path)
        outputs_name = [meta.name for meta in session.get_outputs()]
        return [session.run(outputs_name, feed) for feed in feeds]

    @staticmethod
    def optimize(graph: BaseGraph, knowledge: KnowledgeBase) -> Tuple[bool, BaseGraph]:
        '''Optimize a graph with specific knowledge.'''
        graph_opt = deepcopy(graph)
        res = GraphOptimizer.optimize(graph_opt, knowledge)
        return res, graph_opt

    def check_optimization(self, cfg: OptimizationConfig, expect: bool) -> bool:
        '''Perferm optimization with the provided config, check if the result is as expected.'''
        if expect:
            return self._check_optimization_success(cfg)
        else:
            return self._check_optimization_failure(cfg)

    def check_precision(
        self,
        onnx_ori: str,
        onnx_opt: str,
        feeds: List[Dict[str, NDArray[Any]]],
        cos_th: float = 1e-5,
        atol: float = 1e-5,
        rtol: float = 1e-5
    ) -> bool:
        '''Check inference precision of two graph.'''
        outs_ori = self.inference(onnx_ori, feeds)
        outs_opt = self.inference(onnx_opt, feeds)
        for out_ori, out_opt in zip(outs_ori, outs_opt):
            if len(out_ori) != len(out_opt):
                return False
            if not all(
                meet_precision(lmat, rmat, cos_th=cos_th, rtol=rtol, atol=atol)
                for lmat, rmat in zip(out_ori, out_opt)
            ):
                return False
        return True

    def _check_optimization_success(self, cfg: OptimizationConfig) -> bool:
        success, graph_opt = self.optimize(cfg.graph, cfg.knowledge)
        if not success or self.graph_equal(cfg.graph, graph_opt):
            return False
        if not isinstance(cfg.knowledge, KnowledgeBigKernel):
            success, graph_opt_2 = self.optimize(graph_opt, cfg.knowledge)
            if success or not self.graph_equal(graph_opt, graph_opt_2):
                return False
        cfg.graph.save(cfg.onnx_ori)
        graph_opt.save(cfg.onnx_opt)
        return True

    def _check_optimization_failure(self, cfg: OptimizationConfig) -> bool:
        success, graph_opt = self.optimize(cfg.graph, cfg.knowledge)
        return not success and self.graph_equal(cfg.graph, graph_opt)
