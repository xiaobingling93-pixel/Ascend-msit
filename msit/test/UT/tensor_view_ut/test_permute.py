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

from components.tensor_view.ait_tensor_view.operation import PermuteOperation


class TestPermuteOperation(unittest.TestCase):
    tensor = torch.rand(1, 2, 3, 4)

    def test_unequal_dimension(self):
        op = PermuteOperation("(3,4,2)")
        with self.assertRaises(ValueError):
            op.process(self.tensor)

    def test_duplicate(self):
        op = PermuteOperation("(2,3,2,4,0)")
        with self.assertRaises(ValueError):
            op.process(self.tensor)

    def test_not_n1(self):
        op = PermuteOperation("(0,1,2,3,5)")
        with self.assertRaises(ValueError):
            op.process(self.tensor)

    def test_valid(self):
        op = PermuteOperation("(3,1,2,0)")
        assert torch.equal(op.process(self.tensor), self.tensor.permute(3, 1, 2, 0))
