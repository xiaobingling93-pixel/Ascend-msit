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

import sys
import logging
from logging import StreamHandler


def get_logger():
    mylogger = logging.getLogger('analyze_tool_logger')
    mylogger.propagate = False
    mylogger.setLevel(logging.INFO)

    fmt = '%(asctime)s %(levelname)s : %(message)s'
    formatter = logging.Formatter(fmt)
    console_handler = StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(formatter)
    mylogger.addHandler(console_handler)

    return mylogger


logger = get_logger()
