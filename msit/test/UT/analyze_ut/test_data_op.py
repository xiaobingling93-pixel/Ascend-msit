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

from model_evaluation.common.enum import Framework
from model_evaluation.data import OpMap


class TestOpMap(unittest.TestCase):

    def setUp(self) -> None:
        self.onnx_op_map = OpMap.load_op_map(Framework.ONNX)
        self.tf_op_map = OpMap.load_op_map(Framework.TF)
        self.caffe_op_map = OpMap.load_op_map(Framework.CAFFE)

    def test_load_op_map_fail_case(self):
        op_map = None
        try:
            op_map = OpMap.load_op_map(Framework.UNKNOWN)
        except RuntimeError:
            pass
        self.assertIsNone(op_map)

    def test_load_op_map_success_case(self):
        self.assertIsNotNone(self.onnx_op_map)
        self.assertIsNotNone(self.tf_op_map)
        self.assertIsNotNone(self.caffe_op_map)

    def test_map_op_success_case(self):
        inner_op = self.caffe_op_map.map_op('Convolution')
        self.assertEqual(inner_op, 'Conv2D')

        inner_op = self.tf_op_map.map_op('Conv2D')
        self.assertEqual(inner_op, 'Conv2D')

    def test_map_onnx_op_success_case(self):
        inner_op = self.onnx_op_map.map_onnx_op('Add', 11)
        self.assertEqual(inner_op, 'Add')

        inner_op = self.onnx_op_map.map_onnx_op('Add', 0)
        self.assertIsNone(inner_op)


if __name__ == "__main__":
    unittest.main()
