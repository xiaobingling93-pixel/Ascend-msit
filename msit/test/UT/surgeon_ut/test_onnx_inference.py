# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------
import unittest
from queue import Queue
from unittest.mock import patch, MagicMock
import pytest
import numpy as np

from components.debug.surgeon.auto_optimizer.inference_engine.inference.onnx_inference import ONNXInference

 
# Mock classes and methods
class MockONNXInference(ONNXInference):
    def __init__(self):
        super().__init__()
        self.model = "test_model.onnx"
 
    def _get_params(self, cfg):
        return self.model
 

@pytest.fixture(scope="module")
def onnx_inference_instance():
    return MockONNXInference()
 

@pytest.fixture(scope="module", autouse=True)
def mock_external_dependencies():
    with patch("onnxruntime.InferenceSession") as mock_session:
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance
 
        mock_session_instance.get_inputs.return_value = [
            {"name": "input_1", "shape": None, "type": None, "dtype": np.float32},
        ]
        mock_session_instance.get_outputs.return_value = [
            {"name": "output_1", "shape": None, "type": None, "dtype": np.float32},
        ]
        mock_session_instance.run.return_value = [np.array([1.0, 2.0, 3.0])]
 
        with patch("components.debug.common.logger") as mock_logger:
            yield
 

def test_onnx_inference_given_invalid_model_when_get_params_then_raise_error(onnx_inference_instance):
    onnx_inference_instance._get_params = MagicMock(side_effect=Exception("Model param error"))
 
    with pytest.raises(RuntimeError, match="model gets params error"):
        onnx_inference_instance(1, {}, Queue(), Queue())
 

if __name__ == "__main__":
    unittest.main()
