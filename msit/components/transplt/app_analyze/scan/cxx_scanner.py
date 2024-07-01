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
import os
from multiprocessing import Pool
from app_analyze.utils.log_util import logger
from app_analyze.scan.scanner import Scanner


class CxxScanner(Scanner):
    def __init__(self, files, cxx_parser=None):
        super().__init__(files)
        self.cxx_parser = cxx_parser

    def do_scan(self):
        start_time = time.time()
        result = self.exec_without_threads()
        self.porting_results['cxx'] = result
        eval_time = time.time() - start_time

        logger.debug(f'Total time for scanning cxx files is {eval_time}s')

    def exec_without_threads(self):
        result = {}
        count = max(max(os.cpu_count(), len(self.files)), 16)
        pool = Pool(count)
        list_file = []
        for file in self.files:
            list_file.append((self.cxx_parser, file))
        lst = pool.starmap_async(cxx_parser_, list_file)
        pool.close()
        pool.join()
        lst1 = lst.get(timeout=2)
        for file, r in zip(self.files, lst1):
            result[file] = r
        return result


def cxx_parser_(cxx_parser, files):
    p = cxx_parser(files)
    rst_vals = p.parse()
    return rst_vals
