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
import unittest
from unittest.mock import patch, mock_open, MagicMock

import numpy as np

from auto_optimizer.graph_refactor.onnx.graph import OnnxGraph
from auto_optimizer.pattern.knowledges.knowledge_type_cast import KnowledgeTypeCast
from helper import KnowledgeTestHelper, OptimizationConfig


def make_type_cast_model(onnx_name, x: np.ndarray, y: np.ndarray, value_type: np.dtype):
    graph = OnnxGraph(name=onnx_name)
    graph.add_input('X', value_type, x.shape)
    graph.add_input('Y', value_type, y.shape)
    graph.add_output('O_0', value_type, None)
    graph.add_output('O_1', value_type, None)

    concat_value_0 = np.random.randn(*x.shape).astype(value_type)
    concat_value_1 = np.random.randn(*x.shape).astype(value_type)
    mul_value_0 = np.random.randn(*x.shape).astype(value_type)
    graph.add_initializer('Concat_value_0', concat_value_0)
    graph.add_initializer('Concat_value_1', concat_value_1)
    graph.add_initializer('Mul_value_0', mul_value_0)

    graph.add_node('Add0', 'Add', ['X', 'Y'], ['Add_O'])
    graph.add_node('Squeeze0', 'Squeeze', ['Add_O'], ['O_0'])
    graph.add_node('Mul0', 'Mul', ['Mul_value_0', 'Add_O'], ['Mul_O'])
    graph.add_node('Concat0', 'Concat', ['Concat_value_0', 'Concat_value_1', 'Add_O'], ['O_1'], attrs={'axis': 0})
    graph.update_map()

    graph.infer_shape()
    return graph



class TestKnowledgeTypeCast(unittest.TestCase, KnowledgeTestHelper):
    @patch('onnxruntime.InferenceSession')
    @patch('builtins.open', new_callable=mock_open)
    @patch('auto_optimizer.graph_refactor.onnx.graph.OnnxGraph.parse')
    def test_basic_type_cast(self, mock_parse, mock_file, mock_inference_session):
        for value_type in [np.int64, np.float64]:
            x = np.random.randn(10, 10).astype(value_type)
            y = np.random.randn(10, 10).astype(value_type)

            onnx_name = 'type_cast_test'
            onnx_ori = f'onnx/{onnx_name}.onnx'
            onnx_opt = f'onnx/{onnx_name}_optimize.onnx'
            graph = make_type_cast_model(onnx_name, x, y, value_type)
            cfg = OptimizationConfig(
                graph=graph,
                knowledge=KnowledgeTypeCast(),
                onnx_ori=onnx_ori,
                onnx_opt=onnx_opt,
            )
            
            mock_file.return_value.read.return_value = b'mock onnx file content'
            mock_parse.return_value = graph

            mock_session = MagicMock()
            mock_inference_session.return_value = mock_session

            self.assertTrue(self.check_optimization(cfg=cfg, expect=True))
            feeds = [
                {
                    'X': np.random.randn(*x.shape).astype(x.dtype),
                    'Y': np.random.randn(*y.shape).astype(y.dtype),
                }
                for _ in range(10)
            ]
            self.assertTrue(self.check_precision(onnx_ori, onnx_opt, feeds))


if __name__ == '__main__':
    unittest.main()
