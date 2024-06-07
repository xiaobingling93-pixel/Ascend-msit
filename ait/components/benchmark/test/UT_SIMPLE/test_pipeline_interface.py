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
import sys
import logging
import time
import pytest
import numpy as np

from ais_bench.infer.interface import InferSession
from test_common import TestCommonClass

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
    def generate_input(cls, session, number=500):
        ndatas_list = []
        input_descs = session.get_inputs()
        for _ in range(number):
            ndatas = []
            for input_desc in input_descs:
                barray = bytearray(os.urandom(input_desc.realsize))
                ndata = np.frombuffer(barray)
                ndatas.append(ndata)
            ndatas_list.append(ndatas)
        return ndatas_list

    @classmethod
    def check_results(cls, outputs_single, outputs_multi):
        if isinstance(outputs_single, list) and isinstance(outputs_multi, list):
            for out_sin, out_multi in zip(outputs_single, outputs_multi):
                if not cls.check_results(out_sin, out_multi):
                    return False
            return True

        if isinstance(outputs_single, np.ndarray) and isinstance(outputs_multi, np.ndarray):
            return (outputs_single == outputs_multi).all()

        return outputs_multi == outputs_single

    def init(self):
        self.device_id = 0
        self.model_path = TestCommonClass.get_model_static_om_path(1, "resnet50")

    def test_runable(self):
        session = InferSession(self.device_id, self.model_path)
        ndatas_list = self.generate_input(session)

        outputs = session.infer_pipeline(ndatas_list)

    def test_correctness(self):
        session = InferSession(self.device_id, self.model_path)
        ndatas_list = self.generate_input(session)

        outputs_single = []
        for ndatas in ndatas_list:
            outputs_single.append(session.infer(ndatas))
        outputs_multi = session.infer_pipeline(ndatas_list)

        assert self.check_results(outputs_single, outputs_multi) is True

    def test_performance(self):
        session = InferSession(self.device_id, self.model_path)
        ndatas_list = self.generate_input(session)

        start_single = time.time()
        outputs_single = []
        for ndatas in ndatas_list:
            outputs_single.append(session.infer(ndatas))
        end_single = time.time()

        start_multi = time.time()
        outputs_multi = session.infer_pipeline(ndatas_list)
        end_multi = time.time()

        assert end_multi - start_multi < end_single - start_single


if __name__ == '__main__':
    pytest.main(['test_pipeline_interface.py', '-vs'])
