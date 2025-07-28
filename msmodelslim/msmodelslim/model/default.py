# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import re
from pathlib import Path

import torch
from transformers import PreTrainedTokenizerBase, PreTrainedModel, PretrainedConfig

from msmodelslim.app.base.const import DeviceType
from msmodelslim.app.base.model import BaseModel
from msmodelslim.utils.safe_utils import SafeGenerator
from .factory import ModelFactory


@ModelFactory.register("default")
class DefaultModel(BaseModel):
    def __init__(self,
                 model_type: str,
                 ori_path: Path,
                 device: DeviceType = DeviceType.NPU,
                 trust_remote_code: bool = False):
        super().__init__(model_type, ori_path, device, trust_remote_code)

    def _get_model_pedigree(self) -> str:
        model_type = re.match(r'^[a-zA-Z]+', self.type)
        if model_type is None:
            raise ValueError(f"Invalid model_name: {self.type}.")
        return model_type.group().lower()

    def _load_config(self) -> PretrainedConfig:
        return SafeGenerator.get_config_from_pretrained(model_path=str(self.ori))

    def _load_tokenizer(self, trust_remote_code=False) -> PreTrainedTokenizerBase:
        return SafeGenerator.get_tokenizer_from_pretrained(
            model_path=str(self.ori),
            use_fast=False,
            legacy=False,
            trust_remote_code=trust_remote_code)

    def _load_model(self, device: DeviceType = DeviceType.NPU, trust_remote_code: bool = False) -> PreTrainedModel:
        device_map = 'cpu' if device is DeviceType.CPU else 'auto'
        dtype = self.config.torch_dtype if device is DeviceType.NPU else torch.float32

        return SafeGenerator.get_model_from_pretrained(
            model_path=str(self.ori),
            device_map=device_map,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=trust_remote_code)

    def _load_hook(self) -> None:
        pass

    def _persist_hook(self) -> None:
        pass
