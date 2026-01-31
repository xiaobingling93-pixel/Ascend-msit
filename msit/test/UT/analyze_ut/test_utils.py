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

from model_evaluation.common import utils
from model_evaluation.common.enum import Framework


class TestOpMap(unittest.TestCase):

    def test_check_file_security(self):
        cur_dir = os.path.dirname(os.path.realpath(__file__))

        real_dir = os.path.join(cur_dir, 'testdir')
        os.mkdir(real_dir)
        self.assertFalse(utils.check_file_security(real_dir))
        os.removedirs(real_dir)
        self.assertTrue(utils.check_file_security(real_dir))

        real_file = os.path.join(cur_dir, 'test.sh')
        self.assertTrue(utils.check_file_security(real_file))

    def test_get_framework(self):
        model = 'xxx/test.onnx'
        framework = utils.get_framework(model)
        self.assertEqual(framework, Framework.ONNX)

        model = 'xxx/test.pb'
        framework = utils.get_framework(model)
        self.assertEqual(framework, Framework.TF)

        model = 'xxx/test.prototxt'
        framework = utils.get_framework(model)
        self.assertEqual(framework, Framework.CAFFE)

        model = 'xxx/test.txt'
        framework = utils.get_framework(model)
        self.assertEqual(framework, Framework.UNKNOWN)


if __name__ == "__main__":
    unittest.main()
