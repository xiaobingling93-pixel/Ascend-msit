# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

from typing import List

from transformers import PreTrainedTokenizerBase

from msmodelslim.model.adapter_types import AdapterConfig, MappingConfig
from msmodelslim.quant.processor.anti_outlier.smooth_interface import IterSmoothInterface
from msmodelslim.utils.logging import logger_setter
from msmodelslim.utils.security.model import SafeGenerator

from .default import DefaultModelAdapter
from .factory import ModelFactory

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
class Qwen3ModelAdapter(DefaultModelAdapter, IterSmoothInterface):
    def get_adapter_config_for_subgraph(self) -> List[AdapterConfig]:
        adapter_config = []
        for layer_idx in range(self.config.num_hidden_layers):
            # Norm-Linear融合的映射配置1：输入层归一化到QKV投影
            norm_linear_mapping_config1 = MappingConfig(
                source=f"model.layers.{layer_idx}.input_layernorm",  # 第一个LayerNorm
                targets=[f"model.layers.{layer_idx}.self_attn.k_proj",
                         f"model.layers.{layer_idx}.self_attn.q_proj",
                         f"model.layers.{layer_idx}.self_attn.v_proj"]  # 注意力层的QKV投影
            )

            # Norm-Linear融合的映射配置2：后注意力层归一化到MLP投影
            norm_linear_mapping_config2 = MappingConfig(
                source=f"model.layers.{layer_idx}.post_attention_layernorm",  # 第二个LayerNorm
                targets=[f"model.layers.{layer_idx}.mlp.gate_proj",
                         f"model.layers.{layer_idx}.mlp.up_proj"]  # MLP层的门控和上投影
            )

            # OV融合的映射配置（QKV到输出投影）
            ov_mapping_config = MappingConfig(
                source=f"model.layers.{layer_idx}.self_attn.v_proj",  # V投影层
                targets=[f"model.layers.{layer_idx}.self_attn.o_proj"]  # 输出投影层
            )

            # Up-Down融合的映射配置
            up_down_mapping_config = MappingConfig(
                source=f"model.layers.{layer_idx}.mlp.up_proj",  # 上投影层
                targets=[f"model.layers.{layer_idx}.mlp.down_proj"]  # 下投影层
            )

            # 为当前layer添加4个配置
            adapter_config.extend([
                AdapterConfig(
                    subgraph_type="norm-linear",
                    mapping=norm_linear_mapping_config1
                ),
                AdapterConfig(
                    subgraph_type="norm-linear",
                    mapping=norm_linear_mapping_config2
                ),
                AdapterConfig(
                    subgraph_type="ov",
                    mapping=ov_mapping_config
                ),
                AdapterConfig(
                    subgraph_type="up-down",
                    mapping=up_down_mapping_config
                )
            ])
        return adapter_config

    def _get_model_pedigree(self) -> str:
        return 'qwen3'


@ModelFactory.register("Qwen-QwQ-32B")
@logger_setter(subfix='qwq')
class QwqModelAdapter(DefaultModelAdapter):
    def _get_model_pedigree(self) -> str:
        return 'qwq'
