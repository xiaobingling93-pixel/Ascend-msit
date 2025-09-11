# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

from typing import List, Any, Optional, Generator, Tuple

from torch import nn

from msmodelslim.app import DeviceType
from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.core.graph.adapter_types import AdapterConfig, MappingConfig
from msmodelslim.quant.processor.kv_smooth import KVSmoothFusedType, KVSmoothFusedUnit
from msmodelslim.utils.exception import InvalidModelError
from msmodelslim.utils.logging import logger_setter, get_logger
from .common.layer_wise_forward import generated_decoder_layer_visit_func, transformers_generated_forward_func
from .factory import ModelFactory
from .interface_hub import ModelInfoInterface, ModelSlimPipelineInterfaceV0, ModelSlimPipelineInterfaceV1, \
    AnalyzePipelineInterface, KVSmoothFusedInterface, IterSmoothInterface, FlexSmoothQuantInterface
from .transformers import TransformersModel


@ModelFactory.register("Qwen3-8B")
@ModelFactory.register("Qwen3-14B")
@ModelFactory.register("Qwen3-32B")
@logger_setter()
class Qwen3ModelAdapter(TransformersModel,
                        ModelInfoInterface,
                        ModelSlimPipelineInterfaceV0,
                        ModelSlimPipelineInterfaceV1,
                        AnalyzePipelineInterface,
                        KVSmoothFusedInterface,
                        IterSmoothInterface,
                        FlexSmoothQuantInterface,
                        ):
    def get_model_type(self) -> str:
        return self.model_type

    def get_model_pedigree(self) -> str:
        return 'qwen3'

    def load_model(self, device: DeviceType = DeviceType.NPU) -> nn.Module:
        return self._load_model(device)

    def handle_dataset(self, dataset: Any, device: DeviceType = DeviceType.NPU) -> List[Any]:
        return self._get_tokenized_data(dataset, device)

    def handle_dataset_by_batch(self,
                                dataset: Any,
                                batch_size: int,
                                device: DeviceType = DeviceType.NPU) -> List[Any]:
        return self._get_batch_tokenized_data(calib_list=dataset,
                                              batch_size=batch_size,
                                              device=device)

    def init_model(self, device: DeviceType = DeviceType.NPU) -> nn.Module:
        return self._load_model(device)

    def generate_model_visit(self, model: nn.Module, transformer_blocks: Optional[List[Tuple[str, nn.Module]]] = None,
                             ) -> Generator[ProcessRequest, Any, None]:
        yield from generated_decoder_layer_visit_func(model, transformer_blocks)

    def generate_model_forward(self, model: nn.Module, inputs: Any,
                               ) -> Generator[ProcessRequest, Any, None]:
        yield from transformers_generated_forward_func(model, inputs)

    def enable_kv_cache(self, model: nn.Module, need_kv_cache: bool) -> None:
        return self._enable_kv_cache(model, need_kv_cache)

    def get_kvcache_smooth_fused_subgraph(self) -> List[KVSmoothFusedUnit]:
        return [
            KVSmoothFusedUnit(
                attention_name=f"model.layers.{i}.self_attn",
                layer_idx=i,
                fused_from_query_states_name="q_norm",
                fused_from_key_states_name="k_norm",
                fused_type=KVSmoothFusedType.StateViaRopeToNorm
            )
            for i in range(self.config.num_hidden_layers)
        ]

    def get_head_dim(self) -> int:
        if hasattr(self.config, 'head_dim'):
            return self.config.head_dim

        get_logger().warning(f'head_dim is not found in config.json, use hidden_size // num_attention_heads instead')
        if not hasattr(self.config, 'hidden_size'):
            raise InvalidModelError("hidden_size is not found in config.json",
                                    action="Please check the model config.json")
        if not hasattr(self.config, 'num_attention_heads'):
            raise InvalidModelError("num_attention_heads is not found in config.json",
                                    action="Please check the model config.json")
        if self.config.num_attention_heads == 0:
            raise InvalidModelError("num_attention_heads is 0 in config.json, which should be greater than 0",
                                    action="Please check the model config.json")
        return self.config.hidden_size // self.config.num_attention_heads

    def get_num_key_value_groups(self) -> int:
        if not hasattr(self.config, 'num_attention_heads'):
            raise InvalidModelError("num_attention_heads is not found in config.json",
                                    action=f"Please check config.json in {self.model_path}")
        if not hasattr(self.config, 'num_key_value_heads'):
            raise InvalidModelError("num_key_value_heads is not found in config.json",
                                    action=f"Please check config.json in {self.model_path}")
        if self.config.num_key_value_heads == 0:
            raise InvalidModelError("num_key_value_heads is 0 in config.json, which should be greater than 0",
                                    action=f"Please check config.json in {self.model_path}")
        return self.config.num_attention_heads // self.config.num_key_value_heads

    def get_num_key_value_heads(self) -> int:
        if not hasattr(self.config, 'num_key_value_heads'):
            raise InvalidModelError("num_key_value_heads is not found in config.json",
                                    action=f"Please check config.json in {self.model_path}")
        return self.config.num_key_value_heads

    def get_adapter_config_for_subgraph(self) -> List[AdapterConfig]:
        adapter_config = []
        for layer_idx in range(self.config.num_hidden_layers):
            # Norm-Linear的映射配置1：输入层归一化到QKV投影
            norm_linear_mapping_config1 = MappingConfig(
                source=f"model.layers.{layer_idx}.input_layernorm",  # 第一个LayerNorm
                targets=[f"model.layers.{layer_idx}.self_attn.k_proj",
                         f"model.layers.{layer_idx}.self_attn.q_proj",
                         f"model.layers.{layer_idx}.self_attn.v_proj"]  # 注意力层的QKV投影
            )

            # Norm-Linear的映射配置2：后注意力层归一化到MLP投影
            norm_linear_mapping_config2 = MappingConfig(
                source=f"model.layers.{layer_idx}.post_attention_layernorm",  # 第二个LayerNorm
                targets=[f"model.layers.{layer_idx}.mlp.gate_proj",
                         f"model.layers.{layer_idx}.mlp.up_proj"]  # MLP层的门控和上投影
            )

            # OV的映射配置（QKV到输出投影）
            ov_mapping_config = MappingConfig(
                source=f"model.layers.{layer_idx}.self_attn.v_proj",  # V投影层
                targets=[f"model.layers.{layer_idx}.self_attn.o_proj"]  # 输出投影层
            )

            # Up-Down的映射配置
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
