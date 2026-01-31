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

import torch

from components.tensor_view.ait_tensor_view.atb import read_atb_data, write_atb_data


class TestWriteReadAtbData(unittest.TestCase):
    def test_base_case(self):
        tensor = torch.rand(4, 4, 4, 4)

        write_atb_data(tensor, "./test.bin")
        actual_tensor = read_atb_data("./test.bin")
        os.remove("./test.bin")
        assert torch.equal(tensor, actual_tensor)

    def test_with_dtype(self):
        tensor = torch.rand(4, 4, 4, 4, 4, dtype=torch.float16)

        write_atb_data(tensor, "./test-dtype.bin")
        actual_tensor = read_atb_data("./test-dtype.bin")
        os.remove("./test-dtype.bin")
        assert torch.equal(tensor, actual_tensor)

    def test_with_dims(self):
        tensor = torch.rand(4, 4, 4, 4, 4, dtype=torch.float16)
        tensor.reshape(16, 16, 4, 1)

        write_atb_data(tensor, "./test-dims.bin")
        actual_tensor = read_atb_data("./test-dims.bin")
        os.remove("./test-dims.bin")
        assert torch.equal(tensor, actual_tensor)
