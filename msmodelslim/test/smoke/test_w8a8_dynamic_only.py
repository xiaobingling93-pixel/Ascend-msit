#  -*- coding: utf-8 -*-
#  Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#  http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import pytest
import torch
import torch.nn as nn

from msmodelslim.app import DeviceType
from msmodelslim.core.runner.pipeline_interface import PipelineInterface
from msmodelslim.core.runner.pipeline_parallel_runner import PPRunner
from msmodelslim.quant.ir import W8A8DynamicFakeQuantLinear
from .base import SessionTestCaseBase, is_npu_available, SimpleSequentialAdapter


@pytest.mark.parametrize("test_device, test_dtype", [
    pytest.param("cpu", torch.float32),
    pytest.param("npu", torch.float16, marks=pytest.mark.skipif(not is_npu_available(), reason="NPU not available")),
    pytest.param("npu", torch.bfloat16, marks=pytest.mark.skipif(not is_npu_available(), reason="NPU not available")),
])
@pytest.mark.smoke
class TestW8A8DynamicQuantization(SessionTestCaseBase):
    """测试W8A8动态量化功能的单元测试类"""

    def yaml_file_name(self) -> str:
        return "w8a8_dynamic_only.yaml"

    @pytest.fixture(autouse=True)
    def setup_test_model(self):
        """设置测试模型"""
        self.batch_size = 4
        self.hidden_size = 64
        self.intermedia_size = 128

    def create_model(self):
        return nn.Sequential(
            nn.Linear(self.hidden_size, self.intermedia_size),
            nn.LayerNorm([self.intermedia_size]),
            nn.Linear(self.intermedia_size, self.hidden_size, bias=True)
        )

    @property
    def calib_data(self):
        return [torch.randn(self.batch_size, self.hidden_size) for _ in range(2)]

    def test_w8a8_dynamic_quantization_basic(self, test_device, test_dtype):
        """测试基本的W8A8动态量化功能"""

        torch.set_default_dtype(test_dtype)
        device = torch.device(test_device)
        model = self.create_model().to(device)

        linear_names = set()
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                linear_names.add(name)

        # 执行量化：使用 PPRunner
        adapter: PipelineInterface = SimpleSequentialAdapter(model)
        runner = PPRunner(adapter=adapter)
        for cfg in self.service_cfg.process:
            runner.add_processor(cfg)
        dev_type = DeviceType.NPU if test_device == 'npu' else DeviceType.CPU
        runner.run(model=model, calib_data=self.calib_data, device=dev_type)

        # 检查伪量化成功部署
        for name in linear_names:
            assert isinstance(model.get_submodule(name), W8A8DynamicFakeQuantLinear)

        # 检查伪量化推理
        test_input = torch.randn(self.batch_size, self.hidden_size, device=device)
        output = model(test_input)

        # 验证输出形状是否正确
        assert output.shape == (self.batch_size, self.hidden_size)
