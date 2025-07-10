# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from pathlib import Path

from transformers import PreTrainedTokenizerBase

from msmodelslim.app.base.const import DeviceType
from msmodelslim.utils.safe_utils import SafeGenerator
from .default import DefaultModel
from .factory import ModelFactory


@ModelFactory.register("Qwen2.5-7B-Instruct")
@ModelFactory.register("Qwen2.5-32B-Instruct")
@ModelFactory.register("Qwen2.5-72B-Instruct")
@ModelFactory.register("Qwen2.5-Coder-7B-Instruct")
class Qwen25ModelAdapter(DefaultModel):
    def __init__(self, model_type, ori_path: Path, device: DeviceType = DeviceType.NPU,
                 trust_remote_code: bool = False):
        super().__init__(model_type, ori_path, device, trust_remote_code)
    
    def _get_model_pedigree(self) -> str:
        return 'qwen2_5'

    def _load_tokenizer(self, trust_remote_code=False) -> PreTrainedTokenizerBase:
        return SafeGenerator.get_tokenizer_from_pretrained(
            model_path=str(self.ori),
            use_fast=False,
            legacy=False,
            padding_side='left',
            pad_token='<|extra_0|>',
            eos_token='<|endoftext|>',
            trust_remote_code=trust_remote_code)


@ModelFactory.register("Qwen3-8B")
@ModelFactory.register("Qwen3-14B")
@ModelFactory.register("Qwen3-32B")
class Qwen3ModelAdapter(DefaultModel):
    def _get_model_pedigree(self) -> str:
        return 'qwen3'


@ModelFactory.register("Qwen-QwQ-32B")
class QwqModelAdapter(DefaultModel):
    def _get_model_pedigree(self) -> str:
        return 'qwq'
