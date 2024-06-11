# Copyright (c) 2023 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import os
import shutil

from model_evaluation.data import Opp
from model_evaluation.common import Const
from model_evaluation.common.enum import SocType, Engine


class TestOpp(unittest.TestCase):

    def setUp(self) -> None:
        self.ori_env_map = {
            'ASCEND_TOOLKIT_HOME': os.getenv('ASCEND_TOOLKIT_HOME'),
            'ASCEND_OPP_PATH': os.getenv('ASCEND_OPP_PATH'),
        }
        os.environ['ASCEND_TOOLKIT_HOME'] = ''
        os.environ['ASCEND_OPP_PATH'] = ''

        self.real_bin_path = os.path.dirname(Const.FAST_QUERY_BIN)
        if not os.path.exists(self.real_bin_path):
            os.makedirs(self.real_bin_path)
        mock_bin_path = os.path.join('mock', 'bin', 'ms_fast_query.py')
        shutil.copyfile(mock_bin_path, Const.FAST_QUERY_BIN)

        self.cur_dir = os.path.dirname(os.path.realpath(__file__))

    def tearDown(self) -> None:
        for env, val in self.ori_env_map.items():
            if len(val) != 0:
                os.environ[env] = val
            else:
                os.environ.pop(env)

        os.remove(Const.FAST_QUERY_BIN)
        os.removedirs(self.real_bin_path)

    def test_load_opp_fail_case(self):
        opp = None
        try:
            opp = Opp.load_opp(SocType.Ascend310.name, self.cur_dir)
        except RuntimeError:
            pass
        self.assertIsNone(opp)

    def test_load_opp_success_case(self):
        os.environ['ASCEND_TOOLKIT_HOME'] = self.cur_dir
        os.environ['ASCEND_OPP_PATH'] = self.cur_dir

        opp = Opp.load_opp(SocType.Ascend310.name, self.cur_dir)
        self.assertIsNotNone(opp)

    def test_query_op_info_success_case(self):
        os.environ['ASCEND_TOOLKIT_HOME'] = self.cur_dir
        os.environ['ASCEND_OPP_PATH'] = self.cur_dir

        opp = Opp.load_opp(SocType.Ascend310.name, self.cur_dir)
        self.assertIsNotNone(opp)

        op_info = opp.query_op_info('Abs')
        self.assertEqual(op_info.op_type, 'Abs')
        self.assertEqual(op_info.op_engine, Engine.AICORE)

    def test_query_op_info_fail_case(self):
        os.environ['ASCEND_TOOLKIT_HOME'] = self.cur_dir
        os.environ['ASCEND_OPP_PATH'] = self.cur_dir

        opp = Opp.load_opp(SocType.Ascend310.name, self.cur_dir)
        self.assertIsNotNone(opp)

        op_info = opp.query_op_info('NoneOp')
        self.assertEqual(op_info.op_type, '')

    def test_query_ascend310p_op_engine_success_case(self):
        os.environ['ASCEND_TOOLKIT_HOME'] = self.cur_dir
        os.environ['ASCEND_OPP_PATH'] = self.cur_dir

        opp = Opp.load_opp(SocType.Ascend310P.name, self.cur_dir)
        self.assertIsNotNone(opp)

        engine = opp.query_op_engine('SparseSoftmax')
        self.assertEqual(engine, Engine.AICPU)

        engine = opp.query_op_engine('AdjustContrast')
        self.assertEqual(engine, Engine.DVPP)

    def test_query_ascend310_op_engine_success_case(self):
        os.environ['ASCEND_TOOLKIT_HOME'] = self.cur_dir
        os.environ['ASCEND_OPP_PATH'] = self.cur_dir

        opp = Opp.load_opp(SocType.Ascend310.name, self.cur_dir)
        self.assertIsNotNone(opp)

        engine = opp.query_op_engine('Abs')
        self.assertEqual(engine, Engine.AICORE)

        engine = opp.query_op_engine('NoOp')
        self.assertEqual(engine, Engine.HOST_CPU)

    def test_query_ascend310_op_engine_fail_case(self):
        os.environ['ASCEND_TOOLKIT_HOME'] = self.cur_dir
        os.environ['ASCEND_OPP_PATH'] = self.cur_dir

        opp = Opp.load_opp(SocType.Ascend310.name, self.cur_dir)
        self.assertIsNotNone(opp)

        engine = opp.query_op_engine('NoneOp')
        self.assertEqual(engine, Engine.UNKNOWN)


if __name__ == "__main__":
    unittest.main()
