# Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
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

import time

from app_analyze.common.kit_config import KitConfig
from app_analyze.model.project import Project
from app_analyze.scan.sequence.seq_handler import SeqHandler
from app_analyze.scan.sequence.seq_matcher import match_api_seqs
from app_analyze.scan.sequence.acc_libs import get_expert_libs, set_expert_libs
from app_analyze.scan.sequence.seq_desc import get_idx_tbl, set_api_lut
from app_analyze.solution.seq_advisor import SeqAdvisor
from app_analyze.utils.io_util import IOUtil
from app_analyze.utils.log_util import logger


class SeqProject(Project):
    def __init__(self, inputs, infer_flag=True):
        super().__init__(inputs)
        self.infer_flag = infer_flag
        self._load_api_def(infer_flag)

        self.api_seqs = []

    @staticmethod
    def _load_api_def(flag):
        if not flag:
            logger.debug("Train mode, api lut had been inited outside.")
            return

        all_idx_dict = {}
        for val in KitConfig.API_INDEX_MAP.values():
            idx_seq_dict = IOUtil.json_safe_load(val)
            all_idx_dict.update(idx_seq_dict)
        set_api_lut(all_idx_dict)

        expert_libs = IOUtil.json_safe_load(KitConfig.EXPERT_LIBS_FILE)
        set_expert_libs(expert_libs)

    def _register_reporter_format(self, fmt_dict):
        for reporter in self.reporters:
            name = type(reporter).__name__
            reporter.set_format(fmt_dict.get(name, None))

    def scan(self):
        """
        调用定义的所有扫描器的scan函数进行扫描任务，核心并行扫描处理框架
        在这个函数里面
        :return: NA
        """
        if self.scanners is None:
            raise ValueError('Scanners is none')

        start_time = time.time()
        for scanner in self.scanners:
            scanner.do_scan()
            if scanner.porting_results is not None:
                self.scan_results.update(scanner.porting_results)

        val_dict = self.scan_results.get('cxx', None)
        if not val_dict:
            return

        # handle results
        rst = list()
        for _, seqs in val_dict.items():
            rst += seqs

        if len(val_dict) > 1:
            SeqHandler.union_api_seqs(rst)

        self.api_seqs = SeqHandler.clean_api_seqs(rst, self.infer_flag)
        if self.infer_flag:
            expert_libs = get_expert_libs()
            cluster_result = match_api_seqs(self.api_seqs, expert_libs)
            advisor = SeqAdvisor(cluster_result, get_idx_tbl())
            rd_rst = advisor.recommend()

            self.report_results.update(rd_rst)
            output_format = dict(zip(rd_rst.keys(), [advisor.cxx_format_fn for _ in range(len(rd_rst))]))

            val_dict = self.scan_results.get('cmake', None)
            if val_dict:
                self.report_results.update(val_dict)
                output_format.update(zip(val_dict.keys(), [advisor.common_format_fn for _ in range(len(val_dict))]))

            self._register_reporter_format({'CsvReport': output_format})

        eval_time = time.time() - start_time
        KitConfig.PROJECT_TIME = eval_time

    def get_api_seqs(self):
        return self.api_seqs
