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

from typing import Optional, List, Any

from torch import nn as nn

from msmodelslim import logger
from msmodelslim.core.base.runner import BaseRunner
from msmodelslim.core.runner.layer_wise_runner import LayerWiseRunner
from msmodelslim.core.runner.legacy_runner import LegacyRunner
from msmodelslim.model.default import BaseModelAdapter
from msmodelslim.quant.processor.base import AutoSessionProcessor, AutoProcessorConfig


def process_model(
        model: nn.Module,
        process_cfgs: List[AutoProcessorConfig],
        calib_data: Optional[List[Any]] = None,
        runner: Optional[BaseRunner] = None,
        adapter: Optional[BaseModelAdapter] = None
) -> None:
    logger.info(f"[Session] Quant model with cfg: {process_cfgs}")

    model_adapter = adapter
    runner = create_runner(model, model_adapter) if runner is None else runner

    processors = [AutoSessionProcessor.from_config(model, cfg, model_adapter) for cfg in process_cfgs]

    for processor in processors:
        if not processor.is_data_free() and calib_data is None:
            raise ValueError(f"[Session] Calib data is needed because {processor} is not data-free")
        stage_data = None if processor.is_data_free() else calib_data
        runner.add_processor(processor, stage_data)

    runner.run()


def create_runner(model: nn.Module, model_adapter: BaseModelAdapter) -> BaseRunner:
    try:
        if model_adapter is not None:
            model_adapter.get_decoder_layers()
            return LayerWiseRunner(model)
        return LegacyRunner(model)
    except NotImplementedError:
        logger.warning(f"[Session] Can't create layer wise runner, use legacy runner instead")
        return LegacyRunner(model)
    except Exception as e:
        raise e
