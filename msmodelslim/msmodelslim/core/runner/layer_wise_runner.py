#  -*- coding: utf-8 -*-
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
from typing import List, Optional, Any

from torch import nn

from msmodelslim.app import DeviceType
from msmodelslim.core.runner.generated_runner import GeneratedRunner
from msmodelslim.core.runner.model_hook_interface import ModelHookInterface
from msmodelslim.quant.processor import LoadProcessorConfig, AutoProcessorConfig
from msmodelslim.quant.processor.common.module_func import ModuleFuncProcessorConfig
from msmodelslim.utils.logging import logger_setter


@logger_setter()
class LayerWiseRunner(GeneratedRunner):
    """
        "Module-Processor-Dataset" Mode Runner
        WARNING: Module will be REMOVED after handing, do NOT yield module more than once

        Calibrate quant perception:
        1、Only perceive the processors in front
        2、Perceive the calibration results of all processors in front
    """

    def preprocess_processor(self, processor_list: List[AutoProcessorConfig], model: nn.Module,
                             device: DeviceType = DeviceType.NPU):
        if isinstance(self.adapter, ModelHookInterface):
            processor_list.insert(
                0,
                ModuleFuncProcessorConfig(
                    name="load_state_dict_hook",
                    func=self.adapter.load_state_dict_hook,
                ))

        # 逐层上传推理设备
        processor_list.insert(0,
                              LoadProcessorConfig(
                                  device=device.value, mode="load", post_offload=True
                              ))
        # 逐层释放推理设备内存
        processor_list.append(LoadProcessorConfig(device="meta", mode="offload", cleanup=True))

    def run(self, model: nn.Module = None, calib_data: Optional[List[Any]] = None,
            device: DeviceType = DeviceType.NPU):
        # 强制下放内存，减少显存占用
        super().run(model=model, calib_data=calib_data, device=DeviceType.CPU)
