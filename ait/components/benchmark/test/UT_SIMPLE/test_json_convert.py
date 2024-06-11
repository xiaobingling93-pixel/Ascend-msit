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
import json
import sys
import stat
import logging

import pytest
from ais_bench.infer.benchmark_process import get_legal_json_content

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class TestClass:
    @classmethod
    def setup_class(cls):
        """
        class level setup_class
        """
        cls.init(TestClass)

    @classmethod
    def teardown_class(cls):
        logger.info('\n ---class level teardown_class')

    @classmethod
    def generate_acl_json(cls, json_path, json_dict):
        OPEN_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        OPEN_MODES = stat.S_IWUSR | stat.S_IRUSR
        with os.fdopen(os.open(json_path, OPEN_FLAGS, OPEN_MODES), 'w') as f:
            json.dump(json_dict, f, indent=4, separators=(", ", ": "), sort_keys=True)
        os.chmod(json_path, 0o750)

    def init(self):
        pass

    def test_acl_json_using_msprof(self):
        output_json_dict = {
            "profiler": {
                "switch": "on",
                "aicpu": "on",
                "output": "testdata/profiler",
                "aic_metrics": "",
                "sys_hardware_mem_freq": "50",
                "sys_interconnection_freq": "50",
                "dvpp_freq": "50",
            }
        }
        os.environ.pop('AIT_NO_MSPROF_MODE', None)
        json_path = os.path.realpath("acl_test.json")
        self.generate_acl_json(json_path, output_json_dict)
        cmd_dict = get_legal_json_content(json_path)
        assert cmd_dict.get("--sys-hardware-mem") == "on"
        assert cmd_dict.get("--sys-interconnection-profiling") == "on"
        assert cmd_dict.get("--dvpp-profiling") == "on"


if __name__ == '__main__':
    pytest.main(['test_json_convert.py', '-vs'])
