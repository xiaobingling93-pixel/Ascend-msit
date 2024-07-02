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

import logging

logging.raiseExceptions = False


class Scanner:
    """
    Scanner类作为扫描器的基类存在，仅定义必要的属性和方法接口。
    """
    __slots__ = ['files', 'porting_results', 'name', 'pool_numbers']

    def __init__(self, files):
        self.files = files
        self.porting_results = {}

    def do_scan(self):
        raise NotImplementedError('{} must implement do_scan method!'.format(self.__class__))
