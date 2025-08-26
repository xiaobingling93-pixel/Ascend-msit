#  -*- coding: utf-8 -*-
#  Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
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

from enum import Enum
from typing import List, Optional, Dict, Any

import torch
from torch import nn
from msmodelslim.core.QAL.qtypes import SmoothContext
from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.model.adapter_types import AdapterConfig
from msmodelslim.model.adapter_types import MappingConfig, FusionConfig, SubgraphInfo
from msmodelslim.quant.processor.base import AutoSessionProcessor, AutoProcessorConfig
from msmodelslim.utils.config_map import ConfigSet
from msmodelslim.utils.logging import get_logger


class GraphOpt:
    @staticmethod
    def set_module(model,
                   submodule_key,
                   module):
        tokens = submodule_key.split('.')
        sub_tokens = tokens[:-1]
        cur_mod = model
        for s in sub_tokens:
            cur_mod = getattr(cur_mod, s)
        setattr(cur_mod, tokens[-1], module)


class StatKey(str, Enum):
    STAT_KEY_MAX = "max"
    STAT_KEY_MIN = "min"
    STAT_KEY_SHIFT = "shift"
    STAT_KEY_THRESHOLD_CHANNEL = "thres_c"
    STAT_KEY_THRESHOLD_TENSOR = "thres_t"
    STAT_KEY_SMOOTH_SCALE_MASK = "smooth_scale_mask"
    STAT_KEY_SMOOTH_SCALE = "smooth_scale"
    STAT_KEY_VARIANCE = "std"
    TENSOR = 'tensor'


class VirtualVModule(nn.Module):
    """虚拟 V 模块，用于处理 QKV 融合的情况，支持 MHA、MQA 和 GQA 三种结构"""

    def __init__(self, qkv_module: nn.Linear, num_attention_heads: int, num_key_value_heads: int):
        super().__init__()
        self.qkv_module = qkv_module
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        # 确定注意力机制类型
        self.attention_type = self._determine_attention_type()

        # 计算 V 部分的权重和偏置
        self._extract_v_weights()

    def forward(self, x):
        """前向传播，只返回 V 部分的输出"""
        # 计算 V 部分的输出
        v_output = torch.nn.functional.linear(x, self.v_weight, self.v_bias)
        return v_output

    def update_qkv_weights(self):
        """将更新后的 V 权重更新回原始的 QKV 模块"""
        qkv_weight = self.qkv_module.weight
        qkv_bias = getattr(self.qkv_module, 'bias', None)

        # 计算每个头的维度
        head_dim = qkv_weight.shape[1] // self.num_attention_heads

        # 根据注意力类型计算 V 部分的起始和结束索引
        v_start, v_end = self._get_v_indices(head_dim)

        # 更新 V 部分的权重
        with torch.no_grad():
            qkv_weight[v_start:v_end] = self.v_weight.data

            # 更新 V 部分的偏置
            if qkv_bias is not None and self.v_bias is not None:
                qkv_bias[v_start:v_end] = self.v_bias.data

    def _determine_attention_type(self) -> str:
        """确定注意力机制类型"""
        if self.num_key_value_heads == 1:
            return "MQA"  # Multi-Query Attention
        elif self.num_key_value_heads == self.num_attention_heads:
            return "MHA"  # Multi-Head Attention
        elif (self.num_key_value_heads < self.num_attention_heads and 
              self.num_attention_heads % self.num_key_value_heads == 0):
            return "GQA"  # Grouped-Query Attention
        else:
            get_logger().warning("InValid attention type, please check.")
            return "UNKNOWN"

    def _get_v_indices(self, head_dim: int) -> tuple:
        """根据注意力类型获取 V 部分的索引范围"""
        if self.attention_type == "MHA":
            # MHA: QKV 顺序为 [Q, K, V]，每个都有 num_attention_heads 个头
            q_size = self.num_attention_heads * head_dim
            k_size = self.num_attention_heads * head_dim
            v_start = q_size + k_size
            v_end = q_size + k_size + self.num_attention_heads * head_dim

        elif self.attention_type == "MQA":
            # MQA: QKV 顺序为 [Q, K, V]，Q 有 num_attention_heads 个头，K/V 只有 1 个头
            q_size = self.num_attention_heads * head_dim
            k_size = 1 * head_dim
            v_start = q_size + k_size
            v_end = q_size + k_size + 1 * head_dim

        elif self.attention_type == "GQA":  # GQA
            # GQA: QKV 顺序为 [Q, K, V]，Q 有 num_attention_heads 个头，K/V 有 num_key_value_heads 个头
            q_size = self.num_attention_heads * head_dim
            k_size = self.num_key_value_heads * head_dim
            v_start = q_size + k_size
            v_end = q_size + k_size + self.num_key_value_heads * head_dim
        else:
            raise ValueError(f"Invalid attention type: {self.attention_type}")
        return v_start, v_end

    def _extract_v_weights(self):
        """从 QKV 模块中提取 V 部分的权重和偏置"""
        qkv_weight = self.qkv_module.weight
        qkv_bias = getattr(self.qkv_module, 'bias', None)

        # 计算每个头的维度
        head_dim = qkv_weight.shape[1] // self.num_attention_heads

        # 根据注意力类型获取 V 部分的索引范围
        v_start, v_end = self._get_v_indices(head_dim)

        # 提取 V 部分的权重
        self.v_weight = nn.Parameter(qkv_weight[v_start:v_end].clone())

        # 提取 V 部分的偏置
        if qkv_bias is not None:
            self.v_bias = nn.Parameter(qkv_bias[v_start:v_end].clone())
        else:
            self.v_bias = None


class SubgraphProcessor:
    """
    通用子图适配器，支持通过语义化pattern和scope限定作用域自动查找子图。
    """

    def __init__(
            self,
            model: nn.Module,
            adapter_config: Optional[List[AdapterConfig]] = None,
            m4_config: Optional[AutoProcessorConfig] = None
    ):
        self.adapter_config = adapter_config
        self.m4_config = m4_config
        self.model = model

    def find_subgraphs_by_config(
            self,
            subgraphs: List[SubgraphInfo],
            config: AutoProcessorConfig,
            scope: str,
            **kwargs
    ) -> List[SubgraphInfo]:
        """
        根据配置过滤子图

        Args:
            subgraphs: 子图信息列表
            config: 处理器配置
            scope: 作用域前缀
            **kwargs: 其他参数

        Returns:
            List[SubgraphInfo]: 过滤后的子图信息列表
        """

        result = []
        layer_prefix = f"{scope}." if scope != "" else ""

        include = ConfigSet(self.m4_config.include) if self.m4_config.include else ConfigSet(["*"])
        exclude = ConfigSet(self.m4_config.exclude) if self.m4_config.exclude else ConfigSet([])
        for subgraph in subgraphs:
            # 1. 检查子图类型是否支持
            if subgraph.subgraph_type not in config.enable_subgraph_type:
                continue

            # 2. 检查元数据是否存在
            if not subgraph.metadata or 'source_name' not in subgraph.metadata:
                continue

            source_name = subgraph.metadata['source_name']

            # 3. 检查是否以layer_prefix为前缀
            if not source_name.startswith(layer_prefix):
                continue

            if source_name not in include:
                continue

            if source_name in exclude:
                continue

            result.append(subgraph)

        return result

    def get_global_subgraph_info(self) -> List[SubgraphInfo]:
        """
        获取全局子图信息，优先使用子类实现

        Returns:
            List[SubgraphInfo]: 子图信息列表
        """
        result = []
        if not self.adapter_config:
            raise ValueError("adapter_config cannot be empty")
        result = self._get_adapter_based_subgraph_info()
        return result

    def _get_adapter_based_subgraph_info(self) -> List[SubgraphInfo]:
        """
        基于adapter_config获取子图信息

        Returns:
            List[SubgraphInfo]: 子图信息列表
        """
        result = []

        # 子图类型到创建方法的映射
        subgraph_creators = {
            "norm-linear": lambda m: self._create_norm_linear_subgraph_info(m),
            "ov": lambda m, fusion=None: self._create_ov_subgraph_info(m, fusion),
            "up-down": lambda m: self._create_up_down_subgraph_info(m),
            "linear-linear": lambda m: self._create_linear_linear_subgraph_info(m),
        }

        for adapter_config in self.adapter_config:
            subgraph_type = adapter_config.subgraph_type
            mapping = adapter_config.mapping

            if mapping is None or subgraph_type not in subgraph_creators:
                continue

            # 使用映射的方法创建子图信息
            if subgraph_type == "ov":
                fusion = getattr(adapter_config, 'fusion', None)
                subgraph_info = subgraph_creators[subgraph_type](mapping, fusion)
            else:
                subgraph_info = subgraph_creators[subgraph_type](mapping)

            if subgraph_info:
                result.append(subgraph_info)

        return result

    def _create_subgraph_info_base(
            self,
            mapping: 'MappingConfig',
            subgraph_type: str,
            metadata_extra: Optional[Dict[str, Any]] = None
    ) -> Optional['SubgraphInfo']:
        """
        创建子图信息的基础方法

        Args:
            mapping: 映射配置
            subgraph_type: 子图类型
            metadata_extra: 额外的元数据

        Returns:
            Optional[SubgraphInfo]: 子图信息，如果失败则返回None
        """
        try:
            source_name = mapping.source
            target_names = mapping.targets

            # 获取源模块
            source_module = None
            for name, module in self.model.named_modules():
                if name == source_name:
                    source_module = module
                    break

            if source_module is None:
                get_logger().warning(f"Cannot find source module: {source_name}")
                return None

            # 获取目标模块列表
            target_modules = []
            for target_name in target_names:
                target_module = None
                for name, module in self.model.named_modules():
                    if name == target_name:
                        target_module = module
                        break

                if target_module is not None:
                    target_modules.append(target_module)
                else:
                    get_logger().warning(f"Cannot find target module: {target_name}")

            if not target_modules:
                get_logger().warning(f"No valid target modules found for source: {source_name}")
                return None

            # 构建基础元数据
            metadata = {
                'source_name': source_name,
                'source_module': source_module,
                'target_names': target_names,
                'target_modules': target_modules,
            }

            # 添加额外的元数据
            if metadata_extra:
                metadata.update(metadata_extra)

            return SubgraphInfo(
                name=f"{source_name}_{subgraph_type}_subgraph",
                subgraph_type=subgraph_type,
                metadata=metadata
            )

        except Exception as e:
            get_logger().warning(f"Failed to create {subgraph_type} subgraph info: {e}")
            return None

    def _create_norm_linear_subgraph_info(
            self,
            mapping: 'MappingConfig',
    ) -> Optional['SubgraphInfo']:
        """创建norm-linear子图信息"""
        subgraph_info = self._create_subgraph_info_base(mapping, "norm-linear")
        if subgraph_info is None:
            raise ValueError(
                "Failed to create norm-linear subgraph info. "
                "Please check if the required modules exist in the model "
                "and the mapping configuration is correct."
            )
        return subgraph_info

    def _create_ov_subgraph_info(
            self,
            mapping: 'MappingConfig',
            fusion: Optional['FusionConfig'] = None
    ) -> Optional['SubgraphInfo']:
        """创建ov子图信息"""
        # 准备额外的元数据
        metadata_extra = None
        if fusion:
            metadata_extra = {
                'fusion_type': fusion.fusion_type,
                'num_attention_heads': fusion.num_attention_heads,
                'num_key_value_heads': fusion.num_key_value_heads
            }

        subgraph_info = self._create_subgraph_info_base(mapping, "ov", metadata_extra)
        if subgraph_info is None:
            raise ValueError(
                "Failed to create ov subgraph info. "
                "Please check if the required modules exist in the model "
                "and the mapping configuration is correct."
            )
        return subgraph_info

    def _create_up_down_subgraph_info(
            self,
            mapping: 'MappingConfig'
    ) -> Optional['SubgraphInfo']:
        """创建up-down子图信息"""
        subgraph_info = self._create_subgraph_info_base(mapping, "up-down")
        if subgraph_info is None:
            raise ValueError(
                "Failed to create up-down subgraph info. "
                "Please check if the required modules exist in the model "
                "and the mapping configuration is correct."
            )
        return subgraph_info

    def _create_linear_linear_subgraph_info(
            self,
            mapping: 'MappingConfig'
    ) -> Optional['SubgraphInfo']:
        """创建linear-linear子图信息"""
        subgraph_info = self._create_subgraph_info_base(mapping, "linear-linear")
        if subgraph_info is None:
            raise ValueError(
                "Failed to create linear-linear subgraph info. "
                "Please check if the required modules exist in the model "
                "and the mapping configuration is correct."
            )
        return subgraph_info


class BaseSmoothProcessor(AutoSessionProcessor):

    def __init__(self, model: nn.Module, config: AutoProcessorConfig, adapter: Optional[Any] = None):
        super().__init__(model)
        self.adapter = adapter
        self.config = config
        self.act_stats = {}

    def get_num_attention_heads(self):
        num_attention_heads = None
        key_attention_heads = ["num_attention_heads", "n_head", "num_heads", "heads_num"]
        for key in key_attention_heads:
            if hasattr(self.model.config, key):
                num_attention_heads = getattr(self.model.config, key)
        if not num_attention_heads:
            raise ValueError(
                f"the config of model must have num_attention_heads, n_head or num_heads, \
                                please check or modify the config file"
            )
        return num_attention_heads

    def get_num_key_value_heads(self):
        # 如果不存在num_key_value_heads，则设置为num_attention_heads的值
        if not hasattr(self.model.config, "num_key_value_heads"):
            num_key_value_heads = self.model.config.num_attention_heads
            get_logger().warning("Failed to obtain `num_key_value_heads`, assuming Multi-head Attention by default.")
        else:
            num_key_value_heads = self.model.config.num_key_value_heads

        if not num_key_value_heads:
            raise ValueError(
                f"the config of model must have num_key_value_heads, \
                                please check or modify the config file"
            )
        return num_key_value_heads

    def is_data_free(self) -> bool:
        _ = self
        return False

    def preprocess(self, request: BatchProcessRequest) -> None:
        self._install_statis_hook(request.name, request.module)

    def postprocess(self, request: BatchProcessRequest) -> None:
        self._apply_smooth(request.name, request.module)

    def _build_smooth_context(self, linear_modules: List[nn.Linear]) -> SmoothContext:
        """
        构建 SmoothContext
        
        Args:
            linear_modules: 线性模块列表，用于获取激活统计信息
            
        Returns:
            SmoothContext: 平滑上下文
        """

        for linear_module in linear_modules:
            # 获取模块名称
            module_name = None
            for name, module in self.model.named_modules():
                if module is linear_module:
                    module_name = name
                    break

            if module_name is None:
                get_logger().warning(f"Cannot find module name for {linear_module}")
                continue

            # 获取激活统计信息
            if module_name in self.act_stats:
                stats = self.act_stats[module_name]

                # 获取 smooth_scale
                if StatKey.STAT_KEY_SMOOTH_SCALE in stats:
                    a_smooth_scale = stats[StatKey.STAT_KEY_SMOOTH_SCALE]

                # 获取 shift
                if StatKey.STAT_KEY_SHIFT in stats:
                    shift = stats[StatKey.STAT_KEY_SHIFT]
                    # 创建权重 smooth_scale（初始化为全1）
        w_smooth_scale = torch.ones_like(a_smooth_scale)

        # 创建扩展信息
        ext = {
            'alpha': self.config.alpha,
            'module_names': [name for name, _ in self.model.named_modules() if _ in linear_modules]
        }

        # 创建 SmoothContext
        smooth_context = SmoothContext(
            version=1,
            a_smooth_scale=a_smooth_scale,
            w_smooth_scale=w_smooth_scale,
            shift=shift,
            ext=ext
        )

        return smooth_context

    def _install_statis_hook(self, name: str, module: nn.Module) -> None:
        pass

    def _apply_smooth(self, name: str, module: nn.Module) -> None:
        pass
