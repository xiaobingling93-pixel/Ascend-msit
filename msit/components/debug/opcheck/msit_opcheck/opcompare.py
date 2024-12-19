# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
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

from collections import namedtuple

import torch

from components.debug.common.logger import logger
from components.utils.cmp_algorithm import CMP_ALG_MAP, CUSTOM_ALG_MAP

FLOAT_EPSILON = torch.finfo(torch.float).eps
NAMEDTUPLE_PRECISION_METRIC = namedtuple('precision_metric', ['abs', 'kl', 'cos_sim'])('abs', 'kl', 'cos_sim')
NAMEDTUPLE_PRECISION_MODE = namedtuple(
    'precision_mode', ["keep_origin_dtype", "force_fp16", "force_fp32"]
)("keep_origin_dtype", "force_fp16", "force_fp32")


class OpCompare:
    def __init__(self, case_info=None):
        self.case_info = case_info
        self.case_info['res_detail'] = []

        error1 = 'Error0.1‰'
        error2 = 'Error0.5‰'
        error3 = 'Error1‰'
        error4 = 'Error4‰'
        error5 = 'Error5‰'
        error6 = 'Error+/-1'

        self.precision_standard = {
            'torch.double': [error1, 99.99], 'torch.uint32': [error1, 99.99], 'torch.int64': [error1, 99.99],
            'torch.float32': [error1, 99.99], 'torch.int32': [error1, 99.99], 'torch.uint64': [error1, 99.99],
            'torch.float16': [error3, 99.9], 'torch.bfloat16': [error4, 99.6], 'torch.int8': [error6, 99.9],
            'torch.uint8': [error6, 99], 'torch.int16': [error6, 99.9], 'torch.uint16': [error6, 99.9],
            'torch.bool': [error1, 100]
        }

        self.erol_dict = {
            error1: 0.0001,
            error2: 0.0005,
            error3: 0.001,
            error4: 0.004,
            error5: 0.005,
            error6: 1
        }

    @staticmethod
    def get_rel_pass_rate(out, golden, etol):
        out, golden = out.reshape(-1).cpu(), golden.reshape(-1).cpu()
        size = out.shape[0]
        rel_errors = torch.where(
            torch.abs(golden) > FLOAT_EPSILON,
            torch.abs(out / golden - 1),  # abs(aa - bb) / abs(bb) -> abs(aa / bb - 1)
            torch.tensor(0, dtype=out.dtype),
        )
        rel_pass_rate = torch.sum(rel_errors <= etol) / size if size != 0 else 0
        max_rel_error = torch.max(rel_errors)
        return rel_pass_rate.item() * 100, max_rel_error.item()

    @staticmethod
    def get_abs_pass_rate(out, golden, etol):
        size = out.shape[0]
        abs_errors = torch.where(
            torch.abs(golden) > FLOAT_EPSILON,
            torch.abs(out - golden),  # abs(aa - bb) / abs(bb) -> abs(aa / bb - 1)
            torch.tensor(0, dtype=out.dtype),
        )
        abs_pass_rate = torch.sum(abs_errors <= etol) / size if size != 0 else 0
        max_abs_error = torch.max(abs_errors)
        return abs_pass_rate.item() * 100, max_abs_error.item()

    def get_other_precisions(self, out, golden, etol):
        message = []
        precision_metric = self.case_info['precision_metric']
        default_str = 'NaN'
        abs_pass_rate, max_abs_error, cos_sim, kl = None, None, None, None

        out, golden = out.reshape(-1).cpu().double(), golden.reshape(-1).cpu().double()
        try:
            if NAMEDTUPLE_PRECISION_METRIC.abs in precision_metric:
                abs_pass_rate, max_abs_error = self.get_abs_pass_rate(out, golden, etol)
        except Exception as e:
            logger_text = f"get_abs_pass_rate error: {e}"
            logger.error(logger_text)
        try:
            if NAMEDTUPLE_PRECISION_METRIC.cos_sim in precision_metric:
                cos_sim, cur_message = CMP_ALG_MAP["cosine_similarity"](golden, out)
                if cur_message:
                    message.append('cos_sim: ' + cur_message)
        except Exception as e:
            logger_text = f"get_cosine_similarity error: {e}"
            logger.error(logger_text)
        try:
            if NAMEDTUPLE_PRECISION_METRIC.kl in precision_metric:
                kl, cur_message = CMP_ALG_MAP["kl_divergence"](golden, out)
                if cur_message:
                    message.append('kl_div: ' + cur_message)
        except Exception as e:
            logger_text = f"get_kl_divergence error: {e}"
            logger.error(logger_text)

        abs_pass_rate_str = "%.16f" % float(abs_pass_rate) if abs_pass_rate is not None else default_str
        max_abs_error_str = "%.16f" % float(max_abs_error) if max_abs_error is not None else default_str
        cos_sim_str = "%.10f" % cos_sim if cos_sim is not None else default_str
        kl_div_str = "%.16f" % kl if kl is not None else default_str

        return (abs_pass_rate_str, max_abs_error_str, cos_sim_str, kl_div_str), ", ".join(message)

    def compare(self, out_tensors, golden_out_tensors):
        message, pass_flag = [], True

        my_data_len, golden_data_len = len(out_tensors), len(golden_out_tensors)
        if my_data_len != golden_data_len:
            pass_flag = False
            logger_text = f"Data count not equal, {my_data_len} != {golden_data_len}. Will compare only partial"
            logger.info(logger_text)

        for out_tensor, golden_out_tensor in zip(out_tensors, golden_out_tensors):
            out_dtype = str(out_tensor.dtype)
            p_s = self.precision_standard.get(out_dtype, [])
            if len(p_s) != 2:
                cur_message = f"{out_dtype} not supported!"
                self.case_info['fail_reason'] = cur_message
                raise RuntimeError(cur_message)

            etol = self.erol_dict.get(p_s[0], 0.001)
            err_rate = p_s[1]
            ps_standard = f"{err_rate}%(error<{etol})"

            rel_pass_rate, max_rel = self.get_rel_pass_rate(out_tensor, golden_out_tensor, etol)

            if err_rate >= rel_pass_rate:
                pass_flag = False
                cur_message = f"relative pass rate: {rel_pass_rate} not met standart: {err_rate}."
                message.append(cur_message)
                logger.debug(cur_message)

            rel_pass_rate = "%.16f" % float(rel_pass_rate)
            max_rel = "%.16f" % float(max_rel)
            (abs_pass_rate, max_abs, cos_sim, kl_div), cur_message = self.get_other_precisions(
                out_tensor, golden_out_tensor, etol
            )
            if cur_message:
                message.append(cur_message)

            cur_result = {
                "precision_standard": ps_standard,
                "rel_pass_rate": rel_pass_rate,
                "max_rel": max_rel,
                "abs_pass_rate": abs_pass_rate,
                "max_abs": max_abs,
                "cos_sim": cos_sim,
                "kl_div": kl_div,
            }
            for name, compare_func in CUSTOM_ALG_MAP.items():
                cur_result[name], cur_message = compare_func(golden_out_tensor, out_tensor)
                if cur_message:
                    message.append(f"{name}: {cur_message}")
            self.case_info['res_detail'].append(cur_result)

            if pass_flag:
                self.case_info['excuted_information'] = 'PASS'

            else:
                self.case_info['excuted_information'] = 'FAILED'
            self.case_info['fail_reason'] = ", ".join(message)
