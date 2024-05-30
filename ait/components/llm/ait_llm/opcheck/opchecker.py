# Copyright (c) 2023-2023 Huawei Technologies Co., Ltd.
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
import re
import json
import queue
import threading
import time
import datetime
from collections import namedtuple
import torch

from ait_llm.common.log import logger
from ait_llm.compare.cmp_algorithm import CUSTOM_ALG_MAP


NAMEDTUPLE_PRECISION_METRIC = namedtuple('precision_metric', ['abs', 'kl', 'cos_sim'])('abs', 'kl', 'cos_sim')
NAMEDTUPLE_PRECISION_MODE = namedtuple(
    'precision_mode', ["keep_origin_dtype", "force_fp16", "force_fp32"]
)("keep_origin_dtype", "force_fp16", "force_fp32")


class OpChecker:
    def __init__(self):
        '''
        cases_info结构：
            'op_id': string
            'op_name': string
            'op_param': dict
            'tensor_path': string
        '''
        self.cases_info = {}
        self.completed_op_id_queue = queue.Queue()
        self.special_cases = ('KvCacheOperation', 'ReshapeAndCacheOperation', 'SelfAttentionOperation')
        self.base_path = ''
        self.pid = None
        self.input = ''
        self.output = ''
        self.output_path = ''
        self.operation_ids = ''
        self.check_ids_string = []
        self.operation_name = None
        self.check_patterns = []
        self.precision_metric = []
        self.precision_mode = NAMEDTUPLE_PRECISION_MODE.keep_origin_dtype
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.atb_rerun = False

    @staticmethod
    def third_party_init():
        import ait_llm
        import ctypes

        # Loading libopchecker.so with ctypes
        lib_opchecker_path = os.environ.get("AIT_OPCHECK_LIB_PATH", "")
        if not lib_opchecker_path:
            lib_path_dir = os.path.dirname(os.path.abspath(ait_llm.__file__))
            lib_opchecker_path = os.path.join(lib_path_dir, "opcheck", "libopchecker.so")

        logger.info(f"lib_opchecker_path is {lib_opchecker_path}")
        if not os.path.exists(lib_opchecker_path):
            logger.error(f"{lib_opchecker_path} not exists, check if ait_llm installed correctly")
            return False

        try:
            ctypes.cdll.LoadLibrary(lib_opchecker_path).RegisterAll()
        except Exception as e:
            logger.error(f"{lib_opchecker_path} loading failed, check if ait_llm installed correctly")
            return False

        # Loading libatb_speed_torch.so with torch
        atb_speed_path = os.getenv('ATB_SPEED_HOME_PATH', "")
        if not atb_speed_path:
            logger.error("ATB_SPEED_HOME_PATH is empty, check if mindie_atb_models configured correctly")
            return False

        libatb_speed_torch_path = os.path.join(atb_speed_path, 'lib', 'libatb_speed_torch.so')
        logger.info(f"libatb_speed_torch_path is {libatb_speed_torch_path}")
        if not os.path.exists(libatb_speed_torch_path):
            logger.error(f"{libatb_speed_torch_path} not exists, check if mindie_atb_models configured correctly")
            return False

        try:
            torch.classes.load_library(libatb_speed_torch_path)
        except Exception as e:
            logger.error(f"{libatb_speed_torch_path} loading failed, check if mindie_atb_models configured correctly")
            return False

        return True

    def get_base_path(self, cur_path):
        dirseg = cur_path.split(os.path.sep)
        if len(dirseg) >= 4 and dirseg[-3] == 'tensors' and dirseg[-4].startswith('ait_dump'):
            try:
                pid = dirseg[-2].split("_")[1]
            except:
                pid = None
            return cur_path, pid
        elif cur_path == os.path.dirname(cur_path):
            return None, None
        else:
            return self.get_base_path(os.path.dirname(cur_path))

    def check_input_legality(self, input_path):
        ret = False
        base_path = None
        pid = None

        input_path = os.path.realpath(input_path)
        if not os.path.exists(input_path):
            logger_text = f"Input path not found: {input_path}"
            logger.error(logger_text)
            return input_path, base_path, pid, ret

        base_path, pid = self.get_base_path(input_path)
        if base_path is None:
            logger_text = f"Input path is not in ait_dump tensors directory: {input_path}"
            logger.error(logger_text)
            return input_path, base_path, pid, ret
        
        ret = True
        return input_path, base_path, pid, ret

    def args_init(self, args):
        import torch_npu

        execution_flag = True

        self.input, self.base_path, self.pid, ret = self.check_input_legality(args.input)
        if not ret:
            execution_flag = False
        
        self.output = os.path.realpath(args.output)
        if not os.path.exists(self.output):
            logger_text = f"Output path not found: {self.output}"
            logger.error(logger_text)
            execution_flag = False

        self.output_path = os.path.join(self.output, f"opcheck_result_{self.timestamp}.xlsx")
        self.operation_ids = args.operation_ids
        if self.operation_ids != '':
            try:
                self.check_ids_string = [x.lower().strip() for x in self.operation_ids.split(',')]
            except ValueError as e:
                logger_text = "Failed to parse operation_ids. Error: {}".format(e)
                logger.error(logger_text)
                execution_flag = False
        self.operation_name = args.operation_name
        if self.operation_name is not None:
            try:
                self.check_patterns = [x.lower().strip() for x in self.operation_name.split(',')]
            except ValueError as e:
                logger_text = "Failed to parse operation_name. Error: {}".format(e)
                logger.error(logger_text)
                execution_flag = False
        self.precision_metric = args.precision_metric
        self.precision_mode = args.precision_mode

        # 指定需要使用的npu设备
        try:
            torch.npu.set_device(torch.device(f"npu:{args.device_id}"))
        except RuntimeError as e:
            logger_text = "Failed to set the device. Device_id: {}".format(args.device_id)
            logger.error(logger_text)
            execution_flag = False

        self.atb_rerun = args.atb_rerun
        if self.atb_rerun:
            execution_flag_res = OpChecker.third_party_init()
            if not execution_flag_res:
                execution_flag = False
            else:
                logger_text = "Rerunning operations in atb to calculate outputs..."
                logger.info(logger_text)
        else:
            logger_text = "Comparing outputs in dump data without rerunning operations in atb..."
            logger.info(logger_text)
        return execution_flag

    def start_test(self, args):
        # 0.初始化
        execution_flag_res = self.args_init(args)
        if not execution_flag_res:
            return

        from ait_llm.opcheck.case_manager import CaseManager
        case_manager = CaseManager(self.completed_op_id_queue)
        
        # 1.遍历tensor_path，将算子信息添加到self.cases_info
        self.walk_tensor_path(self.input)
        logger_text = f"Total {len(self.cases_info)} cases found under path: {self.input}"
        logger.info(logger_text)

        # 2.将self.cases_info中的用例添加到case_manager
        result_info = 'excuted_information'
        for _, case_info in self.cases_info.items():
            if_successed_add_case = case_manager.add_case(case_info)
            if if_successed_add_case:
                case_info[result_info] = 'addition successed'
            else:
                case_info[result_info] = 'addition failed'

        # 3.执行测试用例并提供专家建议
        self.excute_cases(case_manager)

        # 4.写入未添加成功的算子
        for v in self.cases_info.values():
            if v[result_info] == 'addition failed':
                v['res_detail'] = []
                self.write_op_result_to_csv(v)
        logger.info(f"\nOpcheck results saved to: {self.output_path}")

    def parse_op_id_name(self, dirpath):
        basename = os.path.basename(dirpath)
        try:
            op_name = basename.split('_')[-1]
        except IndexError as e:
            logger_text = f"{dirpath} is not a valid tensor dir, please check"
            logger.debug(logger_text)
            op_name = None

        rel_path = os.path.relpath(dirpath, self.base_path)
        try:
            op_id = '_'.join([x.split('_')[0] for x in rel_path.split(os.path.sep)])
        except IndexError as e:
            logger_text = f"{dirpath} is not a valid tensor dir, please check"
            logger.debug(logger_text)
            op_id = None

        return op_id, op_name

    def check_id_range(self, op_id):
        if op_id is None:
            return False
        if self.operation_ids == '':
            return True

        for p in self.check_ids_string:
            ret = re.match("^" + p + "(_[0-9]+){0,20}$", op_id)
            if ret:
                return True
        return False

    def check_name(self, op_name):
        if op_name is None:
            return False
        if self.operation_name is None:
            return True

        for p in self.check_patterns:
            if p in op_name.lower():
                return True
        return False

    def is_exec_node(self, case_info):
        if self.operation_ids == '' and self.operation_name is None:
            return True

        flag1 = self.check_id_range(case_info.get("op_id", None))
        flag2 = self.check_name(case_info.get("op_name", None))
        return flag1 and flag2

    def add_case_to_cases(self, case_info):
        op_name = case_info.get("op_name", None)
        op_id = case_info.get("op_id", None)
        if op_name == 'KvCacheOperation':
            case_info['inplace_idx'] = [2]
            self.cases_info[op_id] = case_info
        elif op_name == 'ReshapeAndCacheOperation':
            case_info['inplace_idx'] = [2, 3]
            self.cases_info[op_id] = case_info
        elif op_name == 'SelfAttentionOperation':
            self.cases_info[op_id] = case_info
        else:
            self.cases_info[op_id] = case_info

    def add_op_info_to_cases_info(self, dirpath):
        tensor_path = os.path.join(dirpath, 'after')

        json_path = os.path.join(dirpath, 'op_param.json')
        try:
            with open(json_path, 'r') as f:
                op_param = json.load(f)
        except Exception as e:
            logger_text = f"Cannot loads json file to json! Json file: {json_path} \n Exception: {e}"
            logger.debug(logger_text)
            return

        op_id, op_name = self.parse_op_id_name(dirpath)
        if op_id is None or op_name is None:
            return

        case_info = {
            'op_id': op_id, 'op_name': op_name, 'op_param': op_param, 'tensor_path': tensor_path,
            'precision_metric': self.precision_metric, 'atb_rerun': self.atb_rerun, 'pid': self.pid, 
            'precision_mode': self.precision_mode
        }

        ret = self.is_exec_node(case_info)
        if ret:
            self.add_case_to_cases(case_info)
        return

    def walk_tensor_path(self, cur_path):
        files_and_dirs = os.listdir(cur_path)
        dirnames, filenames = [], []
        for item in files_and_dirs:
            item_path = os.path.join(cur_path, item)
            if os.path.isdir(item_path):
                dirnames.append(item)
            else:
                filenames.append(item)
        if 'after' in dirnames and 'op_param.json' in filenames:
            self.add_op_info_to_cases_info(cur_path)
        for dirname in dirnames:
            if dirname != 'after':
                self.walk_tensor_path(os.path.join(cur_path, dirname))

    def excute_cases(self, case_manager):
        # 定义监控队列函数
        def watching_queue():
            cases_num = len([1 for v in self.cases_info.values() if v["excuted_information"] == 'addition successed'])
            cases_index = 0
            while cases_index < cases_num:
                time.sleep(0.1)
                if not self.completed_op_id_queue.empty():
                    completed_op_id = self.completed_op_id_queue.get()
                    case_info = self.cases_info.get(completed_op_id, '')
                    if case_info != '':
                        self.write_op_result_to_csv(case_info)
                    cases_index += 1
                    logger_text = f"===============excuted cases:{cases_index}, total cases:{cases_num}================"
                    logger.info(logger_text)

        watching_thread = threading.Thread(target=watching_queue)
        watching_thread.start()
        case_manager.excute_cases()
        watching_thread.join()

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