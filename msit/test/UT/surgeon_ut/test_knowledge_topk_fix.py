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

from auto_optimizer.graph_refactor.onnx.graph import OnnxGraph
from auto_optimizer.pattern.knowledges import KnowledgeTopkFix
from helper import KnowledgeTestHelper, OptimizationConfig


def make_topk_model(onnx_name, x: np.ndarray):
    graph = OnnxGraph(name=onnx_name)
    graph.add_input('input', np.float32, x.shape)
    graph.add_output('output_v', np.float32, (10, ))
    graph.add_output('output_i_0', np.int32, (10, ))
    graph.add_output('output_i_1', np.int64, (10, ))
    graph.add_output('topk_i', np.int64, (10, ))

    graph.add_initializer(
        name='topk_k',
        value=np.array([10], np.int64)
    )
    graph.add_node(
        name='topk_0',
        op_type='TopK',
        inputs=['input', 'topk_k'],
        outputs=['topk_v', 'topk_i'],
        attrs={'axis': 0, 'largest': 1, 'sorted': 1}
    )

    graph.add_node(
        name='relu_0',
        op_type='Relu',
        inputs=['topk_v'],
        outputs=['output_v'],
    )

    graph.add_node(
        name='cast_0',
        op_type='Cast',
        inputs=['topk_i'],
        outputs=['output_i_0'],
        attrs={'to': onnx.TensorProto.INT32}
    )

    graph.add_initializer(
        name='add_init',
        value=np.array(list(range(10)), dtype=np.int64)
    )

    graph.add_node(
        name='add_0',
        op_type='Add',
        inputs=['topk_i', 'add_init'],
        outputs=['output_i_1'],
    )

    graph.update_map()
    graph.infer_shape()
    return graph


class TestKnowledgeTopkFix(unittest.TestCase, KnowledgeTestHelper):
    def test_basic_topk_fix(self):
        input_ = np.random.rand(100).astype(np.float32)

        onnx_name = 'topk_model'
        onnx_ori = f'./{onnx_name}.onnx'
        onnx_opt = f'./{onnx_name}_fixed.onnx'
        graph = make_topk_model(onnx_name, input_)
        cfg = OptimizationConfig(
            graph=graph,
            knowledge=KnowledgeTopkFix(),
            onnx_ori=onnx_ori,
            onnx_opt=onnx_opt,
        )
        self.assertTrue(self.check_optimization(cfg=cfg, expect=True))

    def tearDown(self):
        super().tearDown()
        for filename in os.listdir('.'):
            if filename.startswith('topk_model'):
                os.remove(filename)


if __name__ == '__main__':
    unittest.main()
