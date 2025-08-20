# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import re
from pathlib import Path

import torch
from transformers import PreTrainedTokenizerBase, PreTrainedModel, PretrainedConfig

from msmodelslim.app.base.const import DeviceType
from msmodelslim.app.base.model import BaseModelAdapter
from msmodelslim.utils.exception import InvalidModelError, SchemaValidateError
from msmodelslim.utils.exception_decorator import exception_handler
from msmodelslim.utils.logging import logger_setter
from msmodelslim.utils.security.model import SafeGenerator
from .factory import ModelFactory


@ModelFactory.register("default")
@logger_setter()
class DefaultModelAdapter(BaseModelAdapter):
    @exception_handler('Using default model',
                       err_cls=Exception,
                       ms_err_cls=InvalidModelError,
                       action='Please check the model type')
    def __init__(self,
                 model_type: str,
                 ori_path: Path,
                 device: DeviceType = DeviceType.NPU,
                 trust_remote_code: bool = False):
        super().__init__(model_type, ori_path, device, trust_remote_code)

    def _get_model_pedigree(self) -> str:
        model_type = re.match(r'^[a-zA-Z]+', self.type)
        if model_type is None:
            raise SchemaValidateError(f"Invalid model_name: {self.type}.",
                                      action='Please check the model name')
        return model_type.group().lower()

    def _load_config(self) -> PretrainedConfig:
        return SafeGenerator.get_config_from_pretrained(model_path=str(self.ori))

    def _load_tokenizer(self, trust_remote_code=False) -> PreTrainedTokenizerBase:
        return SafeGenerator.get_tokenizer_from_pretrained(
            model_path=str(self.ori),
            use_fast=False,
            legacy=False,
            trust_remote_code=trust_remote_code)

    def _load_model(self, device_map=None, torch_dtype=None) -> PreTrainedModel:
        device_map = device_map if device_map is not None else self._device_map
        dtype = torch_dtype if torch_dtype is not None else self._torch_dtype

        return SafeGenerator.get_model_from_pretrained(
            model_path=str(self.ori),
            device_map=device_map,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=self._trust_remote_code)

    def _load_hook(self) -> None:
        pass

    def _persist_hook(self) -> None:
        pass
