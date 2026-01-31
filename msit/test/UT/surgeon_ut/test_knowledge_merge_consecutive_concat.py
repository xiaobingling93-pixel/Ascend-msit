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
import unittest

import numpy as np
import onnx
from onnx import (
    helper,
    TensorProto,
)

from auto_optimizer.graph_refactor.onnx.graph import OnnxGraph
from auto_optimizer.pattern.knowledges.knowledge_merge_consecutive_concat import KnowledgeMergeConsecutiveConcat
from helper import KnowledgeTestHelper, OptimizationConfig


def make_c2_concat_model(onnx_name, x, y, z, diff_axis=False):
    input_x = helper.make_tensor_value_info("X", TensorProto.FLOAT, x.shape)
    input_y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, y.shape)
    input_z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, z.shape)

    input_o = helper.make_tensor_value_info("O", TensorProto.FLOAT, None)

    axis = 1 if diff_axis else 0
    node_concat0 = helper.make_node("Concat", ["X", "Y"], ["X_S"], "Concat0", axis=0)
    node_concat1 = helper.make_node("Concat", ["X_S", "Z"], ["O"], "Concat1", axis=axis)

    graph = helper.make_graph(
        [node_concat0, node_concat1], "continue_concat_test",
        [input_x, input_y, input_z], [input_o]
    )
    model = helper.make_model(graph, ir_version=8)

    del model.opset_import[:]
    opset = model.opset_import.add()
    opset.domain = ''
    opset.version = 11
    onnx.save(model, onnx_name)


class TestKnowledgeMergeConsecutiveConcat(unittest.TestCase, KnowledgeTestHelper):

    def test_merge_c2_concat(self):
        x = np.random.randn(2, 1, 2).astype(np.float32)
        y = np.random.randn(1, 1, 2).astype(np.float32)
        z = np.random.randn(3, 1, 2).astype(np.float32)

        onnx_name = "c2_concat"
        onnx_ori = f"./{onnx_name}.onnx"
        onnx_opt = f"./{onnx_name}_optimize.onnx"

        make_c2_concat_model(onnx_ori, x, y, z, False)

        graph = OnnxGraph.parse(onnx_ori)
        cfg = OptimizationConfig(
            graph=graph,
            knowledge=KnowledgeMergeConsecutiveConcat(),
            onnx_ori=onnx_ori,
            onnx_opt=onnx_opt,
        )
        self.assertTrue(self.check_optimization(cfg=cfg, expect=True))
        feeds = [
            {
                'X': np.random.randn(*x.shape).astype(x.dtype),
                'Y': np.random.randn(*y.shape).astype(y.dtype),
                'Z': np.random.randn(*z.shape).astype(z.dtype),
            }
            for _ in range(10)
        ]
        self.assertTrue(self.check_precision(onnx_ori, onnx_opt, feeds))

    def test_merge_c2_diff_axis_concat(self):
        x = np.random.randn(2, 1, 2).astype(np.float32)
        y = np.random.randn(1, 1, 2).astype(np.float32)
        z = np.random.randn(3, 1, 2).astype(np.float32)

        onnx_name = "c2_concat_diff_axis"
        onnx_ori = f"./{onnx_name}.onnx"

        make_c2_concat_model(onnx_ori, x, y, z, True)
        graph = OnnxGraph.parse(onnx_ori)

        cfg = OptimizationConfig(
            graph=graph,
            knowledge=KnowledgeMergeConsecutiveConcat(),
        )
        self.assertTrue(self.check_optimization(cfg=cfg, expect=False))

    def tearDown(self):
        super().tearDown()
        for filename in os.listdir('.'):
            if filename.startswith('c2_concat'):
                os.remove(filename)


if __name__ == "__main__":
    unittest.main()
