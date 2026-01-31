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
from unittest.mock import mock_open, patch, MagicMock

import numpy as np

from auto_optimizer.graph_refactor.onnx.graph import OnnxGraph
from auto_optimizer.pattern.knowledges.knowledge_resize_mode_to_nearest import KnowledgeResizeModeToNearest
from helper import KnowledgeTestHelper, OptimizationConfig


def make_resize_model(onnx_name, x: np.ndarray, y: np.ndarray, value_type: np.dtype):
    graph = OnnxGraph(name=onnx_name)
    graph.add_input('input', value_type, x.shape)
    graph.add_output('11', value_type, None)

    roi = np.random.randn(0).astype(np.int64)
    scales = np.random.randn(0).astype(np.int64)
    graph.add_initializer('roi', roi)
    graph.add_initializer('scales', scales)
    graph.add_node('Resize0', 'Resize', ['input', 'scales'], ['11'], attrs={
        'coordinate_transformation_mode': b"half_pixel",
        'cubic_coeff_a': -0.75,
        'exclude_outside': 0,
        'mode': b"linear",
        'nearest_mode': b"round_prefer_floor",
    })
    graph.update_map()

    graph.infer_shape()
    return graph


class TestKnowledgeResizeModeToNearest(unittest.TestCase, KnowledgeTestHelper):
    @patch('onnxruntime.InferenceSession')
    @patch('builtins.open', new_callable=mock_open)
    @patch('auto_optimizer.graph_refactor.onnx.graph.OnnxGraph.parse')
    def test_basic_resize_mode(self, mock_parse, mock_file, mock_inference_session):
        for value_type in [np.int64]:
            input_x = np.random.randn(10, 10).astype(value_type)
            input_y = np.random.randn(10, 10).astype(value_type)

            onnx_name = 'resize_mode_test'
            onnx_ori = f'onnx/{onnx_name}1.onnx'
            onnx_opt = f'onnx/{onnx_name}_optimize1.onnx'
            graph = make_resize_model(onnx_name, input_x, input_y, value_type)
            cfg = OptimizationConfig(
                graph=graph,
                knowledge=KnowledgeResizeModeToNearest(),
                onnx_ori=onnx_ori,
                onnx_opt=onnx_opt,
            )
            mock_file.return_value.read.return_value = b'mock onnx file content'
            mock_parse.return_value = graph

            mock_session = MagicMock()
            mock_inference_session.return_value = mock_session

            self.assertTrue(self.check_optimization(cfg=cfg, expect=True))


if __name__ == '__main__':
    unittest.main()
