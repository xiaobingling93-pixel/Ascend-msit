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

import numpy as np

from test_node_common import create_node


class TestPlaceHolder(unittest.TestCase):

    def test_placeholder_get_dtype(self):
        ph = create_node('OnnxPlaceHolder')
        self.assertEqual(ph.dtype, np.dtype('float32'))

    def test_placeholder_set_dtype(self):
        ph = create_node('OnnxPlaceHolder')
        ph.dtype = np.dtype('float16')
        self.assertEqual(ph.dtype, np.dtype('float16'))

    def test_placeholder_get_shape(self):
        ph = create_node('OnnxPlaceHolder')
        self.assertEqual(ph.shape, [1, 3, 224, 224])

    def test_placeholder_set_shape(self):
        ph = create_node('OnnxPlaceHolder')
        ph.shape = [-1, 3, 224, 224]
        self.assertEqual(ph.shape, ['-1', 3, 224, 224])


if __name__ == "__main__":
    unittest.main()
