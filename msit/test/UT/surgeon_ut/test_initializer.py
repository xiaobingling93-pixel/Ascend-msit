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


class TestInitializer(unittest.TestCase):

    def test_initializer_get_value(self):
        ini = create_node('OnnxInitializer')
        self.assertTrue(np.array_equal(ini.value, np.array([[1, 2, 3, 4, 5]], dtype='int32'), equal_nan=True))
        self.assertEqual(ini.value.dtype, 'int32')

    def test_initializer_set_value(self):
        ini = create_node('OnnxInitializer')
        ini.value = np.array([[7, 8, 9], [10, 11, 12]], dtype='float32')
        self.assertTrue(np.array_equal(ini.value, np.array([[7, 8, 9], [10, 11, 12]], dtype='float32'), equal_nan=True))
        self.assertEqual(ini.value.dtype, 'float32')


if __name__ == "__main__":
    unittest.main()
