# -*- coding: utf-8 -*-
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
from typing import Dict, Set, List

from model_evaluation.graph.onnx import OnnxGraph
from model_evaluation.common.enum import ONNXCheckerError


class OnnxChecker:
    def __init__(self, graph: OnnxGraph) -> None:
        self._graph = graph

    @staticmethod
    def _check_op_input(graph, err_map):
        nodes_outputs: Set[str] = set()
        for node in graph.node:
            nodes_outputs = nodes_outputs.union(node.output)

        model_inputs: Set[str] = set()
        for input_ in graph.input:
            model_inputs.add(input_.name)

        initializers: Set[str] = set()
        for initializer in graph.initializer:
            initializers.add(initializer.name)

        def is_valid(input_: str) -> bool:
            return input_ in nodes_outputs or input_ in initializers or input_ in model_inputs

        for node in graph.node:
            empty_input = [input_ for input_ in node.input if not is_valid(input_)]
            if len(empty_input) == 0:
                continue
            s = ','.join(empty_input)
            err_map.setdefault(node.name, []).append(f'input {s} is empty.')

    def check_ops(self) -> Dict[str, List[str]]:
        if self._graph is None:
            return {}

        graph = self._graph.graph

        err_map: Dict[str, List[str]] = {}
        self._check_op_validation_by_onnx_checker(graph, err_map)
        self._check_op_input(graph, err_map)
        return err_map

    def _check_op_validation_by_onnx_checker(self, graph, err_map):
        for node in graph.node:
            errcode, errinfo = self._graph.check_node(node)
            if errcode == ONNXCheckerError.UNREGISTERED_OP or errcode == ONNXCheckerError.SUCCESS:
                continue
            err_map.setdefault(node.name, []).append(errinfo)
