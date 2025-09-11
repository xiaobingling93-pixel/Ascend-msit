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

import os
import shutil
import tempfile
from abc import abstractmethod
from functools import lru_cache
from pathlib import Path

import pytest
import torch
import yaml
from torch import nn

from msmodelslim.app import DeviceType
from msmodelslim.app.quant_service.modelslim_v1.quant_config import ModelslimV1ServiceConfig
from msmodelslim.app.quant_service.modelslim_v1.save import AscendV1Config
from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.core.runner.pipeline_interface import PipelineInterface
from msmodelslim.model.base import BaseModelAdapter


class SessionTestCaseBase:
    """pytest版本的测试基类，使用fixture模式替代unittest.TestCase"""

    @pytest.fixture(autouse=True)
    def setup_session_test(self):
        """自动设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_dir = os.path.realpath(self.temp_dir)
        assert os.path.exists(self.temp_dir)

        # 创建W8A8量化配置
        self.yaml_cfg_file = os.path.join(os.path.dirname(__file__), "configs", self.yaml_file_name())
        self.service_cfg = ModelslimV1ServiceConfig.model_validate(
            yaml.safe_load(open(self.yaml_cfg_file, "r"))['service_cfg'])

        yield

        # 清理资源
        shutil.rmtree(self.temp_dir)

    @pytest.fixture(autouse=True)
    def setup_ascendv1(self, setup_session_test):
        """设置AscendV1配置"""
        for save_cfg in self.service_cfg.save:
            save_cfg.set_save_directory(self.temp_dir)
            if isinstance(save_cfg, AscendV1Config):
                save_cfg.part_file_size = 0

    @abstractmethod
    def yaml_file_name(self) -> str:
        raise NotImplementedError(f"You should provide a yaml name to init test session")


@lru_cache(maxsize=1)
def is_npu_available():
    try:
        import torch_npu
        return torch.npu.is_available()
    except ImportError:
        return False


@lru_cache(maxsize=1)
def is_cuda_available():
    try:
        return torch.cuda.is_available()
    except ImportError:
        return False


class SimpleSequentialAdapter(BaseModelAdapter, PipelineInterface):
    """最简适配器：将任意 nn.Module 作为整体进行访问与前向。

    - handle_dataset: 返回 [(tensor,)] 列表，供 forward 使用
    - init_model: 直接返回传入的模型到 device
    - generate_model_visit: 仅针对整体模型发出一次访问请求
    - generate_model_forward: 针对整体模型发出一次前向请求
    - enable_kv_cache: 空实现
    """

    def __init__(self, model: nn.Module):
        super().__init__("", Path(""), True)
        self._model = model

    def handle_dataset(self, dataset, device: DeviceType = DeviceType.NPU):
        if dataset is None:
            return []
        # 测试侧按模型当前设备放置数据，避免依赖被测代码的 device 传递
        try:
            model_dev = next(self._model.parameters()).device
        except StopIteration:
            model_dev = torch.device('cpu')
        processed = []
        for item in dataset:
            # 支持 dict 作为 kwargs 传递
            if isinstance(item, dict):
                kwargs = {}
                for k, v in item.items():
                    if hasattr(v, 'to'):
                        item[k] = v.to(model_dev)

                processed.append(kwargs)
                continue

            if isinstance(item, (list, tuple)):
                # 位置参数场景：若检测到前两个是 input_ids/attention_mask 语义的张量，则转为 long
                converted = []
                for idx, t in enumerate(item):
                    if hasattr(t, 'to'):
                        item[idx] = t.to(model_dev)
                    converted.append(t)
                inputs = tuple(converted)
            else:
                inputs = (item.to(model_dev) if hasattr(item, 'to') else item,)
            processed.append(inputs)
        return processed

    def init_model(self, device: DeviceType = DeviceType.NPU) -> nn.Module:
        dev = str(device).lower()
        return self._model.to(dev)

    def generate_model_visit(self, model: nn.Module, transformer_blocks=None):
        yield ProcessRequest(name="", module=model, args=tuple(), kwargs={})

    def generate_model_forward(self, model: nn.Module, inputs):
        # 若 inputs 为 kwargs（dict），则走 kwargs 形式
        if isinstance(inputs, dict):
            yield ProcessRequest(name="", module=model, args=tuple(), kwargs=inputs)
            return
        yield ProcessRequest(name="", module=model, args=inputs, kwargs={})

    def enable_kv_cache(self, model: nn.Module, need_kv_cache: bool) -> None:
        return None
