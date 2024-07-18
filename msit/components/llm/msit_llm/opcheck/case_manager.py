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

import os
import json
import unittest
import multiprocessing
from msit_llm.opcheck.check_case import OP_NAME_DICT
from msit_llm.compare.cmp_algorithm import CUSTOM_ALG_MAP
from msit_llm.opcheck.opchecker import NAMEDTUPLE_PRECISION_METRIC


class CaseManager:
    def __init__(self, output_path='./', precision_metric=[]):
        self.output_path = output_path
        self.precision_metric = precision_metric
        self.cases = []

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
        cases = []
        for case_info in chunk_cases:
            op = OP_NAME_DICT[case_info['op_name']]
            testloader = unittest.TestLoader()
            testnames = testloader.getTestCaseNames(op)
            for name in testnames:
                op_cur = op(name, case_info=case_info)
                self.cases.append(op_cur)
                suite.addTest(op_cur)
        return suite, cases

    def run_test(self, suite, cases):
        # 拉起测试套
        runner = unittest.TextTestRunner(verbosity=2)
        runner.run(suite)

        # 写入文件
        for case in cases:
            self.write_op_result_to_csv(case.case_info)

    def excute_cases(self, num_processes=1):
        # 多进程执行测试用例
        chunk_size = len(self.cases) // num_processes
        pool = multiprocessing.get_context('spawn')
        processes = []

        for i in range(num_processes):
            start_index = i * chunk_size
            end_index = start_index + chunk_size if i != num_processes - 1 else len(self.cases)
            chunk_cases = self.cases[start_index:end_index]
            suite, cases = self.add_cases_to_suite(chunk_cases)
            process = pool.Process(target=self.run_test, args=(suite, cases))
            processes.append(process)
            process.start()

        for process in processes:
            process.join() # 等待所有子进程完成

    def _update_single_op_result(self, op_info, cur_id, res_detail):
        default_str = 'NaN'
        excuted_information = op_info["excuted_information"]
        required = [
            op_info["op_id"], op_info["op_name"], op_info["op_param"], op_info["tensor_path"],
            cur_id, res_detail.get('precision_standard', default_str), excuted_information,
            res_detail.get('rel_pass_rate', default_str), res_detail.get('max_rel', default_str),
        ]
        if NAMEDTUPLE_PRECISION_METRIC.abs in self.precision_metric:
            required.append(res_detail.get('abs_pass_rate', default_str))
            required.append(res_detail.get('max_abs', default_str))
        if NAMEDTUPLE_PRECISION_METRIC.cos_sim in self.precision_metric:
            required.append(res_detail.get('cos_sim', default_str))
        if NAMEDTUPLE_PRECISION_METRIC.kl in self.precision_metric:
            required.append(res_detail.get('kl_div', default_str))

        custom_ret = [res_detail.get(custom_name, default_str) for custom_name in CUSTOM_ALG_MAP]
        return required + custom_ret + [op_info.get('fail_reason', default_str)]

    def write_op_result_to_csv(self, op_result):
        import openpyxl

        if not os.path.exists(self.output_path):
            wb = openpyxl.Workbook()
            ws = wb.active
            required_head = [
                'op_id', 'op_name', 'op_param', 'tensor_path', 'out_tensor_id', 'precision_standard',
                'precision_result', 'rel_precision_rate(%)', 'max_rel_error'
            ]
            if NAMEDTUPLE_PRECISION_METRIC.abs in self.precision_metric:
                required_head.append('abs_precision_rate(%)')
                required_head.append('max_abs_error')
            if NAMEDTUPLE_PRECISION_METRIC.cos_sim in self.precision_metric:
                required_head.append('cosine_similarity')
            if NAMEDTUPLE_PRECISION_METRIC.kl in self.precision_metric:
                required_head.append('kl_divergence')
            custom_header = list(CUSTOM_ALG_MAP.keys())
            ws.append(required_head + custom_header + ["fail_reason"])
            wb.save(self.output_path)

        wb = openpyxl.load_workbook(self.output_path)
        ws = wb.active

        op_info = {
            "op_id": op_result.get('op_id', ""),
            "op_name": op_result.get('op_name', ""),
            "op_param": json.dumps(op_result.get('op_param', "")),
            "tensor_path": op_result.get('tensor_path', ""),
            "excuted_information": op_result.get('excuted_information', ""),
            "fail_reason": op_result.get('fail_reason', ""),
        }
        
        if len(op_result['res_detail']) > 0:
            for cur_id, res_detail in enumerate(op_result['res_detail']):
                ws.append(self._update_single_op_result(op_info, cur_id, res_detail))
        else:
            cur_id, res_detail = 'NaN', {}
            ws.append(self._update_single_op_result(op_info, cur_id, res_detail))
        wb.save(self.output_path)