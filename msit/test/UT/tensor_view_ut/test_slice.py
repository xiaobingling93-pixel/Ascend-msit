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

import torch

from components.tensor_view.ait_tensor_view.operation import SliceOperation


class TestSliceOperation(unittest.TestCase):
    def test_single_dimension(self):
        tensor = torch.rand(4, 4, 4)
        op1 = SliceOperation("[3]")
        op2 = SliceOperation("[:3]")
        op3 = SliceOperation("[1:2:]")
        op4 = SliceOperation("[::2]")
        assert torch.equal(op1.process(tensor), tensor[3])
        assert torch.equal(op2.process(tensor), tensor[:3])
        assert torch.equal(op3.process(tensor), tensor[1:2:])
        assert torch.equal(op4.process(tensor), tensor[::2])

    def test_negative(self):
        tensor = torch.rand(1, 2, 3)
        op1 = SliceOperation("[-1]")
        op2 = SliceOperation("[..., -1:-3]")
        op3 = SliceOperation("[0, -1:-2, -2:-3]")
        assert torch.equal(op1.process(tensor), tensor[-1])
        assert torch.equal(op2.process(tensor), tensor[..., -1:-3])
        assert torch.equal(op3.process(tensor), tensor[0, -1:-2, -2:-3])

    def test_ellipsis(self):
        tensor = torch.rand(3, 4, 5)
        op = SliceOperation("[..., 1]")
        assert torch.equal(op.process(tensor), tensor[..., 1])

    def test_multi_dimension(self):
        tensor = torch.rand(3, 4, 5)
        assert torch.equal(SliceOperation("[1, 2:4, 0]").process(tensor), tensor[1, 2:4, 0])

    def test_invalid_dimension(self):
        tensor = torch.rand(3, 4)
        with self.assertRaises(ValueError):
            op = SliceOperation("[1, 2, 3]")
            op.process(tensor)

    def test_index_out_of_range(self):
        tensor = torch.rand(3, 4)
        with self.assertRaises(IndexError):
            op = SliceOperation("[3, 0]")
            op.process(tensor)
