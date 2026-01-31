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
import os
from copy import deepcopy

import onnx
from onnx.onnx_cpp2py_export.checker import ValidationError
from google.protobuf.message import DecodeError

from model_evaluation.common.enum import ONNXCheckerError
from components.utils.check.rule import Rule


class OnnxGraph:
    def __init__(self, graph) -> None:
        super().__init__()
        self._graph = graph

    @property
    def graph(self):
        return self._graph.graph

    @classmethod
    def load(cls, model_path: str) -> 'OnnxGraph':
        if not os.path.isfile(model_path):
            raise RuntimeError(f'model {model_path} is not file.')

        try:
            Rule.input_file().check(model_path, will_raise=True)
            graph = onnx.load_model(model_path)
        except DecodeError as e:
            raise RuntimeError(f'{e}') from e

        return cls(graph)

    def opset_version(self) -> int:
        if self._graph is None:
            return -1

        opset_import = self._graph.opset_import
        if len(opset_import) == 0:
            return -1
        return opset_import[0].version

    def check_node(self, node):
        check_ctx = onnx.checker.DEFAULT_CONTEXT
        ori_opset_imports = deepcopy(check_ctx.opset_imports)
        check_ctx.opset_imports = {'': self.opset_version()}

        try:
            onnx.checker.check_node(node, check_ctx)
        except ValidationError as e:
            if str(e).find("No Op registered") != -1:
                return ONNXCheckerError.UNREGISTERED_OP, str(e)
            return ONNXCheckerError.OTHERS, str(e)

        check_ctx.opset_imports = ori_opset_imports
        return ONNXCheckerError.SUCCESS, ""
