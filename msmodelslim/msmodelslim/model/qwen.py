# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

from transformers import PreTrainedTokenizerBase

from msmodelslim.utils.security.model import SafeGenerator
from .default import DefaultModelAdapter
from .factory import ModelFactory
from ..utils.logging import logger_setter


@ModelFactory.register("Qwen2.5-7B-Instruct")
@ModelFactory.register("Qwen2.5-32B-Instruct")
@ModelFactory.register("Qwen2.5-72B-Instruct")
@ModelFactory.register("Qwen2.5-Coder-7B-Instruct")
@logger_setter(subfix='qwen2_5')
class Qwen25ModelAdapter(DefaultModelAdapter):
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
@logger_setter(subfix='qwen3')
class Qwen3ModelAdapter(DefaultModelAdapter):
    def _get_model_pedigree(self) -> str:
        return 'qwen3'


@ModelFactory.register("Qwen-QwQ-32B")
@logger_setter(subfix='qwq')
class QwqModelAdapter(DefaultModelAdapter):
    def _get_model_pedigree(self) -> str:
        return 'qwq'
