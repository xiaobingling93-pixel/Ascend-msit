# Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
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
from msit_llm.opcheck.check_case import OP_NAME_DICT


class CaseManager:
    def __init__(self, excuted_ids=None):
        self.suite = unittest.TestSuite()
        self.cases = []
        self.excuted_ids = excuted_ids

    def add_case(self, case_info):
        op_name = case_info['op_name']
        if op_name not in OP_NAME_DICT.keys():
            #没有该op_name
            return False 
        else:
            if OP_NAME_DICT[op_name]:
                self.cases.append(case_info)
                return True
            else:
                #该算子用例未添加
                return False
      
    def add_cases_to_suite(self):
        for case_info in self.cases:
            op = OP_NAME_DICT[case_info['op_name']]
            self.suite.addTest(op.parametrize(optest_class=op, case_info=case_info, excuted_ids=self.excuted_ids))

    def excute_cases(self):
        self.add_cases_to_suite()

        #拉起测试套
        runner = unittest.TextTestRunner(verbosity=2)
        runner.run(self.suite)