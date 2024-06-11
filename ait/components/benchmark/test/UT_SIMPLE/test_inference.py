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
    def get_input_tensor_name():
        return "actual_input_1"

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

    def test_infer_runcase_dict(self):
        device_id = 0
        options = aclruntime.session_options()
        model_path = TestCommonClass.get_model_static_om_path(1, self.model_name)
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
        logger.info(session.sumary())

    def test_infer_runcase_list(self):
        device_id = 0
        options = aclruntime.session_options()
        model_path = TestCommonClass.get_model_static_om_path(1, self.model_name)
        session = aclruntime.InferenceSession(model_path, device_id, options)

        # create new numpy data according inputs info
        barray = bytearray(session.get_inputs()[0].realsize)
        ndata = np.frombuffer(barray)
        # convert numpy to pytensors in device
        tensor = aclruntime.Tensor(ndata)
        tensor.to_device(device_id)

        outnames = [session.get_outputs()[0].name]
        feeds = [tensor]

        outputs = session.run(outnames, feeds)
        logger.info("outputs:", outputs)

        for out in outputs:
            out.to_host()
        logger.info(session.sumary())

    def test_infer_runcase_empty_outputname(self):
        device_id = 0
        options = aclruntime.session_options()
        model_path = TestCommonClass.get_model_static_om_path(1, self.model_name)
        session = aclruntime.InferenceSession(model_path, device_id, options)

        # create new numpy data according inputs info
        barray = bytearray(session.get_inputs()[0].realsize)
        ndata = np.frombuffer(barray)
        # convert numpy to pytensors in device
        tensor = aclruntime.Tensor(ndata)
        tensor.to_device(device_id)

        outnames = []
        feeds = [tensor]

        outputs = session.run(outnames, feeds)
        logger.info("outputs:", outputs)

        for out in outputs:
            out.to_host()
        logger.info(session.sumary())

    def test_infer_runcase_none_outputname(self):
        device_id = 0
        options = aclruntime.session_options()
        model_path = TestCommonClass.get_model_static_om_path(1, self.model_name)
        session = aclruntime.InferenceSession(model_path, device_id, options)

        # create new numpy data according inputs info
        barray = bytearray(session.get_inputs()[0].realsize)
        ndata = np.frombuffer(barray)
        # convert numpy to pytensors in device
        tensor = aclruntime.Tensor(ndata)
        tensor.to_device(device_id)

        outnames = None
        feeds = [tensor]

        with pytest.raises(TypeError) as e:
            outputs = session.run(outnames, feeds)
            logger.info("outputs:", outputs)

    def test_infer_runcase_split(self):
        device_id = 0
        options = aclruntime.session_options()
        model_path = TestCommonClass.get_model_static_om_path(1, self.model_name)
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
        logger.info(session.sumary())

    def test_infer_runcase_split_list(self):
        device_id = 0
        options = aclruntime.session_options()
        model_path = TestCommonClass.get_model_static_om_path(1, self.model_name)
        session = aclruntime.InferenceSession(model_path, device_id, options)

        # create new numpy data according inputs info
        barray = bytearray(session.get_inputs()[0].realsize)
        ndata = np.frombuffer(barray)
        # convert numpy to pytensors in device
        tensor = aclruntime.Tensor(ndata)
        tensor.to_device(device_id)

        outnames = [session.get_outputs()[0].name]
        feeds = [tensor]

        outputs = session.run(outnames, feeds)
        logger.info("outputs:", outputs)

        for out in outputs:
            out.to_host()
        logger.info(session.sumary())

    def test_infer_invalid_input_size(self):
        device_id = 0
        options = aclruntime.session_options()
        model_path = TestCommonClass.get_model_static_om_path(1, self.model_name)
        session = aclruntime.InferenceSession(model_path, device_id, options)

        # create new numpy data according inputs info
        barray = bytearray(session.get_inputs()[0].realsize + 128)
        ndata = np.frombuffer(barray)
        # convert numpy to pytensors in device
        tensor = aclruntime.Tensor(ndata)
        tensor.to_device(device_id)

        outnames = [session.get_outputs()[0].name]
        feeds = {session.get_inputs()[0].name: tensor}

        with pytest.raises(RuntimeError) as e:
            outputs = session.run(outnames, feeds)
            logger.info("outputs:", outputs)

    def test_infer_invalid_input_type(self):
        device_id = 0
        options = aclruntime.session_options()
        model_path = TestCommonClass.get_model_static_om_path(1, self.model_name)
        session = aclruntime.InferenceSession(model_path, device_id, options)

        # create new numpy data according inputs info
        barray = bytearray(session.get_inputs()[0].realsize)
        ndata = np.frombuffer(barray)

        outnames = [session.get_outputs()[0].name]
        feeds = {session.get_inputs()[0].name: ndata}

        with pytest.raises(TypeError) as e:
            outputs = session.run(outnames, feeds)
            logger.info("outputs:", outputs)

    def test_infer_invalid_outname(self):
        device_id = 0
        options = aclruntime.session_options()
        model_path = TestCommonClass.get_model_static_om_path(1, self.model_name)
        session = aclruntime.InferenceSession(model_path, device_id, options)

        # create new numpy data according inputs info
        barray = bytearray(session.get_inputs()[0].realsize)
        ndata = np.frombuffer(barray)
        # convert numpy to pytensors in device
        tensor = aclruntime.Tensor(ndata)
        tensor.to_device(device_id)

        outnames = [session.get_outputs()[0].name + "xxx"]
        feeds = {session.get_inputs()[0].name: tensor}

        with pytest.raises(RuntimeError) as e:
            outputs = session.run(outnames, feeds)
            logger.info("outputs:", outputs)

    def test_infer_invalid_device_id(self):
        device_id = 0
        options = aclruntime.session_options()
        model_path = TestCommonClass.get_model_static_om_path(1, self.model_name)
        session = aclruntime.InferenceSession(model_path, device_id, options)

        # create new numpy data according inputs info
        barray = bytearray(session.get_inputs()[0].realsize)
        ndata = np.frombuffer(barray)
        # convert numpy to pytensors in device
        tensor = aclruntime.Tensor(ndata)
        with pytest.raises(RuntimeError) as e:
            tensor.to_device(device_id + 100)
