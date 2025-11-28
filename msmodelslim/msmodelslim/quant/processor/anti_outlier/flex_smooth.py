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

"""Flex Smooth Quantization Processor (Refactored)"""

from typing import Callable, Any, Literal, Annotated, List, Optional, Dict

import torch
import torch.distributed as dist
import torch.nn as nn
from pydantic import AfterValidator, Field

from msmodelslim.core.QAL.qregistry import QABCRegistry
from msmodelslim.quant.processor.anti_outlier.common import (
    FlexSmoothQuantConfig,
    FlexAWQSSZConfig,
    FlexSmoothQuantContext,
    FlexAWQSSZContext
)
from msmodelslim.quant.processor.anti_outlier.api.smooth_api import flex_smooth_quant, flex_awq_ssz
from msmodelslim.quant.processor.anti_outlier.smooth_base import BaseSmoothProcessor, BaseSmoothProcessorConfig
from msmodelslim.quant.processor.anti_outlier.smooth_interface import FlexSmoothQuantInterface
from msmodelslim.quant.processor.anti_outlier.common.smooth_components import StatsCollector, SubgraphRegistry, StatKey
from msmodelslim.quant.processor.base import AutoSessionProcessor
from msmodelslim.quant.quantizer.linear import LinearQConfig
from msmodelslim.quant.observer import MsMinMaxObserver, MinMaxObserverConfig
from msmodelslim.utils.exception import UnsupportedError
from msmodelslim.utils.logging import get_logger, logger_setter
from msmodelslim.utils.validation.value import validate_normalized_value, is_string_list


class FlexSmoothBaseProcessorConfig(BaseSmoothProcessorConfig):
    """Base configuration class for Flex processors, defining common fields and validation rules"""
    type: Literal["_abstract_flex_smooth_base_"] = "_abstract_flex_smooth_base_"
    
    alpha: Annotated[float, AfterValidator(validate_normalized_value)] = None
    beta: Annotated[float, AfterValidator(validate_normalized_value)] = None
    enable_subgraph_type: Annotated[list, AfterValidator(is_string_list)] = Field(
        default_factory=lambda: ["norm-linear", "linear-linear", "ov", "up-down"]
    )


class FlexSmoothQuantProcessorConfig(FlexSmoothBaseProcessorConfig):
    """FlexSmoothQuant processor configuration"""
    type: Literal["flex_smooth_quant"] = "flex_smooth_quant"


class FlexAWQSSZProcessorConfig(FlexSmoothBaseProcessorConfig):
    """FlexAWQSSZ processor configuration"""
    type: Literal["flex_awq_ssz"] = "flex_awq_ssz"
    qconfig: LinearQConfig


class FlexStatsCollector(StatsCollector):
    """
    Flex smooth statistics collector
    """
    
    def __init__(self):
        super().__init__()
        # 为每个模块名称创建observer，用于收集channel_max统计
        self.observers: Dict[str, MsMinMaxObserver] = {}
    
    def create_hook(self, name: str, subgraph_type: str = None) -> Callable:
        def stats_hook(module: nn.Linear, input_tensor: tuple, output: Any) -> None:
            # 有的路由专家可能采集不到激活，需要跳过
            if not input_tensor or not isinstance(input_tensor, tuple):
                get_logger().warning(f"Input tensor is empty for module {name}")
                return

            tensor = input_tensor[0]
            hidden_dim = tensor.shape[-1]
            tensor = tensor.reshape(-1, hidden_dim).detach()

            if name not in self.act_stats:
                self.act_stats[name] = {}
            statis_dict = self.act_stats[name]

            # 收集tensor用于后续算法
            cpu_tensor = tensor.to("cpu").reshape(-1, tensor.shape[-1])
            if StatKey.TENSOR not in statis_dict:
                statis_dict[StatKey.TENSOR] = [cpu_tensor]
            else:
                statis_dict[StatKey.TENSOR].append(cpu_tensor)

            if name not in self.observers:
                observer_config = MinMaxObserverConfig(dim=0, keepdim=False)
                self.observers[name] = MsMinMaxObserver(observer_config)
            abs_tensor = tensor.abs()
            self.observers[name].update(abs_tensor)
            _, channel_max = self.observers[name].get_min_max()
            statis_dict[StatKey.STAT_KEY_SMOOTH_SCALE] = channel_max

        return stats_hook
    
    def clear_stats(self) -> None:
        super().clear_stats()
        for observer in self.observers.values():
            observer.reset()
        self.observers.clear()


class FlexSmoothBaseProcessor(BaseSmoothProcessor):    
    def __init__(self, model: nn.Module, config: FlexSmoothBaseProcessorConfig, adapter: object, **kwargs):
        super().__init__(model, config, adapter)
        if not isinstance(adapter, FlexSmoothQuantInterface):
            raise UnsupportedError(
                f'{adapter.__class__.__name__} does not implement FlexSmoothQuantInterface',
                action='Please provide a valid model adapter which implements FlexSmoothQuantInterface'
            )
        
        self.config = config
        self._validate_parameters()
        self.stats_collector = FlexStatsCollector()


@QABCRegistry.register(dispatch_key=FlexSmoothQuantProcessorConfig, abc_class=AutoSessionProcessor)
@logger_setter()
class FlexSmoothQuantProcessor(FlexSmoothBaseProcessor):
    """FlexSmoothQuant Processor"""
    
    def apply_smooth_algorithm(self, subgraph_obj: Any, linear_names: List[str]) -> None:
        """Apply FlexSmoothQuant algorithm"""
        subgraph_type = SubgraphRegistry.get_name(type(subgraph_obj))
        config = FlexSmoothQuantConfig(
            alpha=self.config.alpha,
            beta=self.config.beta,
            extra_config=getattr(subgraph_obj, 'extra_config', None)
        )
        smooth_context = self._build_smooth_context(linear_names)
        if smooth_context is None:
            get_logger().warning(
                "No statistics collected for %s subgraph, skipping. "
                "This may happen for unused MOE experts.",
                subgraph_type
            )
            return
        flex_smooth_quant(subgraph_obj, config, smooth_context)
        get_logger().info(f"Successfully applied FlexSmoothQuant to {subgraph_type} subgraph")

    def _build_smooth_context(self, linear_names: List[str]) -> Optional[FlexSmoothQuantContext]:
        a_smooth_scale = None
        tensors = None
        if not linear_names:
            get_logger().warning(
                "No linear modules provided while building FlexSmoothQuantContext; skipping smooth application."
            )
            return None
        # 仅用第一个linear的激活统计信息
        linear_name = linear_names[0]

        # 获取激活统计信息
        if linear_name in self.stats_collector.act_stats:
            stats = self.stats_collector.act_stats[linear_name]

            # 获取 smooth_scale
            if StatKey.STAT_KEY_SMOOTH_SCALE in stats:
                a_smooth_scale = stats[StatKey.STAT_KEY_SMOOTH_SCALE]
            else:
                a_smooth_scale = None

            # 获取 tensors
            if StatKey.TENSOR in stats:
                tensors = stats[StatKey.TENSOR]
            else:
                tensors = None
        else:
            get_logger().warning(f"Linear name {linear_name} not in act_stats")
            return None

        # 检查是否成功获取到激活平滑尺度
        if a_smooth_scale is None or tensors is None:
            # 返回 None 而不是抛出异常，让调用者决定如何处理
            get_logger().debug(
                "Failed to get activation smooth scale from linear name %s. "
                "This may happen for unused subgraphs (e.g., unactivated MOE experts).",
                linear_name
            )
            return None
        # 创建 FlexSmoothQuantContext
        smooth_context = FlexSmoothQuantContext(
            version=1,
            a_smooth_scale=a_smooth_scale,
            tensors=tensors,
        )

        return smooth_context


@QABCRegistry.register(dispatch_key=FlexAWQSSZProcessorConfig, abc_class=AutoSessionProcessor)
@logger_setter()
class FlexAWQSSZProcessor(FlexSmoothBaseProcessor):
    """FlexAWQSSZ Processor"""
    
    def apply_smooth_algorithm(self, subgraph_obj: Any, linear_names: List[str]) -> None:
        """Apply FlexAWQSSZ algorithm"""
        subgraph_type = SubgraphRegistry.get_name(type(subgraph_obj))
        config = FlexAWQSSZConfig(
            alpha=self.config.alpha,
            beta=self.config.beta,
            qconfig=self.config.qconfig
        )
        smooth_context = self._build_smooth_context(linear_names)
        if smooth_context is None:
            get_logger().warning(
                "No statistics collected for %s subgraph, skipping. "
                "This may happen for unused MOE experts.",
                subgraph_type
            )
            return
        flex_awq_ssz(subgraph_obj, config, smooth_context)
        get_logger().info(f"Successfully applied FlexAWQSSZ to {subgraph_type} subgraph")

    def _build_smooth_context(self, linear_names: List[str]) -> Optional[FlexAWQSSZContext]:
        tensors = None
        if not linear_names:
            get_logger().warning(
                "No linear modules provided while building FlexSmoothQuantContext; skipping smooth application."
            )
            return None
        # 仅用第一个linear的激活统计信息
        linear_name = linear_names[0]

        # 获取激活统计信息
        if linear_name in self.stats_collector.act_stats:
            stats = self.stats_collector.act_stats[linear_name]

            # 获取 tensors
            if StatKey.TENSOR in stats:
                tensors = stats[StatKey.TENSOR]
            else:
                tensors = None
        else:
            get_logger().warning(f"Linear name {linear_name} not in act_stats")
            return None

        # 检查是否成功获取到激活平滑尺度
        if tensors is None:
            # 返回 None 而不是抛出异常，让调用者决定如何处理
            get_logger().debug(
                "Failed to get activation tensors from linear name %s. "
                "This may happen for unused subgraphs (e.g., unactivated MOE experts).",
                linear_name
            )
            return None
        # 创建 FlexAWQSSZContext
        smooth_context = FlexAWQSSZContext(
            version=1,
            tensors=tensors
        )

        return smooth_context
