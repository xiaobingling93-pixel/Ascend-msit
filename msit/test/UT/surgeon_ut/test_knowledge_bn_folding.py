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
from numpy.typing import NDArray

from auto_optimizer.graph_refactor.onnx.graph import OnnxGraph
from auto_optimizer.pattern.knowledges import KnowledgeBNFolding
from helper import KnowledgeTestHelper, OptimizationConfig


def var_channel(arr: NDArray) -> NDArray:
    for i in reversed(range(len(arr.shape))):
        if i == 1:
            continue
        arr = arr.var(axis=i, keepdims=False)
    return arr


def mean_channel(arr: NDArray) -> NDArray:
    for i in reversed(range(len(arr.shape))):
        if i == 1:
            continue
        arr = arr.mean(axis=i, keepdims=False)
    return arr


def make_bn_folding_graph(name, input_: NDArray, perm, attrs, const) -> OnnxGraph:
    assert len(input_.shape) > 2, 'invalid input'
    graph = OnnxGraph(name=name)
    graph.add_input('input', input_.dtype, input_.shape)
    graph.add_output('output', input_.dtype, input_.shape)

    data = input_.transpose(perm)
    shape_ = [data.shape[1]]

    # input -> TR0 -> BN -> TR1 -> output
    graph.add_node(
        name='tr0',
        op_type='Transpose',
        inputs=['input'],
        outputs=['input_tr'],
        attrs={'perm': perm}
    )
    graph.add_initializer(
        name='bn_scale',
        value=np.random.rand(*shape_).astype(data.dtype) / 2 + 0.5,
    )
    if const:
        graph.add_initializer(
            name='bn_mean',
            value=mean_channel(data) + np.random.randn(*shape_).astype(data.dtype) / 100,
        )
    else:
        graph.add_node(
            name='mean_input',
            op_type='ReduceMean',
            inputs=['input'],
            outputs=['bn_mean'],
            attrs={'axes': [i for i in range(len(data.shape)) if i != 1], 'keepdims': 0}
        )
    graph.add_initializer(
        name='bn_bias',
        value=np.random.rand(*shape_).astype(data.dtype) / 2 + 0.5,
    )
    graph.add_initializer(
        name='bn_var',
        value=var_channel(data) + np.random.randn(*shape_).astype(data.dtype) / 1e8,
    )
    graph.add_node(
        name='bn0',
        op_type='BatchNormalization',
        inputs=['input_tr', 'bn_scale', 'bn_bias', 'bn_mean', 'bn_var'],
        outputs=['output_tr'],
        attrs=attrs
    )
    graph.add_node(
        name='tr1',
        op_type='Transpose',
        inputs=['output_tr'],
        outputs=['output'],
        attrs={'perm': perm}
    )

    graph.update_map()
    graph.infer_shape()
    return graph


class TestKnowledgeBNFolding(unittest.TestCase, KnowledgeTestHelper):
    def test_bn_folding(self):
        test_cases = [
            ('2x3x4_021_t0_e5_c1_f32', True, [2, 3, 4], [0, 2, 1], {}, True, np.float32),
            ('2x3x4_021_t0_e5_c1_f64', True, [2, 3, 4], [0, 2, 1], {}, True, np.float64),
            ('10x12x14_021_t0_e5_c1_f32', True, [10, 12, 14], [0, 2, 1], {}, True, np.float32),
            ('10x12x14_021_t0_e6_c1_f32', True, [10, 12, 14], [0, 2, 1], {'epsilon': 1e-6}, True, np.float32),
            ('2x3x4x5_0132_t0_e4_c1_f32', True, [2, 3, 4, 5], [0, 1, 3, 2], {'epsilon': 1e-4}, True, np.float32),
            ('2x3x4x5_0132_t0_e0_c1_f32', True, [2, 3, 4, 5], [0, 1, 3, 2], {'epsilon': 1.0}, True, np.float32),
            ('2x3x4x5x6_04231_t0_e2_c1_f32', True, [2, 3, 4, 5, 6], [0, 4, 2, 3, 1], {'epsilon': 1e-2},
             True, np.float32),

            # 非常数输入，无法优化
            ('2x3x4_021_t0_e5_c0_f32', False, [2, 3, 4], [0, 2, 1], {}, False, np.float32),
            # training_mode为1，无法优化
            ('2x3x4_021_t1_e5_c1_f32', False, [2, 3, 4], [0, 2, 1], {'training_mode': 1}, True, np.float32),
            # 前后transpose的perm无法抵消，无法优化
            ('2x3x4x5_0312_t0_e4_c1_f32', False, [2, 3, 4, 5], [0, 3, 1, 2], {'epsilon': 1e-4}, True, np.float32),
        ]
        for name, expect, shape_, perm, attrs, const, dtype in test_cases:
            with self.subTest(name):
                onnx_name = f'bn_folding_{name}'
                onnx_ori = f'./{onnx_name}_ori.onnx'
                onnx_opt = f'./{onnx_name}_opt.onnx'

                input_ = np.random.rand(*shape_).astype(dtype) * 10
                graph = make_bn_folding_graph(onnx_name, input_, perm=perm, attrs=attrs, const=const)

                cfg = OptimizationConfig(
                    graph=graph,
                    knowledge=KnowledgeBNFolding(),
                    onnx_ori=onnx_ori,
                    onnx_opt=onnx_opt,
                )
                self.assertTrue(self.check_optimization(cfg=cfg, expect=expect))
                if not expect:
                    return
                self.assertTrue(self.check_precision(onnx_ori, onnx_opt, feeds=[{'input': input_}]))

    def tearDown(self):
        super().tearDown()
        for filename in os.listdir('.'):
            if filename.startswith('bn_folding_'):
                os.remove(filename)


if __name__ == '__main__':
    unittest.main()
