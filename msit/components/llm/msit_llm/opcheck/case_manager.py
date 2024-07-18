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
import multiprocessing
from msit_llm.opcheck.check_case import OP_NAME_DICT


class CaseManager:
    def __init__(self):
        self.suite = unittest.TestSuite()
        self.cases = []
        self.ops = []

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

    def add_cases_to_suite(self, chunk_cases):
        suite = unittest.TestSuite()
        for case_info in chunk_cases:
            op = OP_NAME_DICT[case_info['op_name']]
            testloader = unittest.TestLoader()
            testnames = testloader.getTestCaseNames(op)
            for name in testnames:
                op_cur = op(name, case_info=case_info)
                self.ops.append(op_cur)
                suite.addTest(op_cur)
        return suite

    def run_test(self, suite):
        #拉起测试套
        runner = unittest.TextTestRunner(verbosity=2)
        runner.run(suite)

    def excute_cases(self, num_processes=1):
        # 多进程执行测试用例
        chunk_size = len(self.cases) // num_processes
        pool = multiprocessing.get_context('spawn')
        processes = []

        for i in range(num_processes):
            start_index = i * chunk_size
            end_index = start_index + chunk_size if i != num_processes - 1 else len(self.cases)
            chunk_cases = self.cases[start_index:end_index]
            suite = self.add_cases_to_suite(chunk_cases)
            process = pool.Process(target=self.run_test, args=(suite,))
            processes.append(process)
            process.start()

        for process in processes:
            process.join() # 等待所有子进程完成