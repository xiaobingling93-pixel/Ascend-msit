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

import aclruntime
import numpy as np
import pytest
from test_common import TestCommonClass

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class TestClass:
    @staticmethod
    def test_args_invalid_model_path():
        device_id = 0
        model_path = "xxx_invalid.om"
        options = aclruntime.session_options()
        with pytest.raises(RuntimeError) as e:
            aclruntime.InferenceSession(model_path, device_id, options)
            logger.info("when om_path invalid error msg is %s", e)

    @classmethod
    def setup_class(cls):
        """
        class level setup_class
        """
        cls.init(TestClass)

    @classmethod
    def teardown_class(cls):
        logger.info('\n ---class level teardown_class')

    def init(self):
        self.model_name = "resnet50"

    def test_args_invalid_device_id(self):
        device_id = 100
        model_path = TestCommonClass.get_model_static_om_path(1, self.model_name)
        options = aclruntime.session_options()
        with pytest.raises(RuntimeError) as e:
            aclruntime.InferenceSession(model_path, device_id, options)
            logger.info("when device_id invalid error msg is %s", e)

    ## 待完善 增加 loopo 和 log_level的校验和判断 当前不完善

    def test_args_ok(self):
        device_id = 0
        model_path = TestCommonClass.get_model_static_om_path(1, self.model_name)
        options = aclruntime.session_options()
        session = aclruntime.InferenceSession(model_path, device_id, options)

        # create new numpy data according inputs info
        barray = bytearray(session.get_inputs()[0].realsize)
        ndata = np.frombuffer(barray)
        # convert numpy to pytensors in device
        tensor = aclruntime.Tensor(ndata)
        tensor.to_device(device_id)

        outnames = [session.get_outputs()[0].name]
        feeds = {session.get_inputs()[0].name: tensor}

        outputs = session.run(outnames, feeds)
        logger.info("outputs:", outputs)

        for out in outputs:
            out.to_host()
        # summary inference throughput
        logger.info(session.sumary())
