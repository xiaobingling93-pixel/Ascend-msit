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

"""
Anti-outlier processor module

This module provides processors for outlier suppression algorithms.
Config classes must be imported immediately for registration with AutoProcessorConfig.
Processor classes are lazily loaded to avoid importing unused algorithms.
"""

__all__ = [
    # Processors
    "IterSmoothProcessor",
    "IterSmoothProcessorConfig",
    "FlexSmoothQuantProcessor",
    "FlexSmoothQuantProcessorConfig",
    "FlexAWQSSZProcessor",
    "FlexAWQSSZProcessorConfig",
    "SubgraphRegistry",
    "HookManager",
    "StatsCollector"
]


from .common.smooth_components import HookManager, StatsCollector, SubgraphRegistry
from .iter_smooth import IterSmoothProcessorConfig
from .impl import iter_smooth

from .flex_smooth import (
    FlexSmoothQuantProcessorConfig,
    FlexAWQSSZProcessorConfig
)
from .impl import flex_smooth_quant, flex_awq_ssz


def __getattr__(name: str):
    """Lazy import for processor classes."""
    if name == "IterSmoothProcessor":
        from .iter_smooth import IterSmoothProcessor
        return IterSmoothProcessor
    elif name == "FlexSmoothQuantProcessor":
        from .flex_smooth import FlexSmoothQuantProcessor
        return FlexSmoothQuantProcessor
    elif name == "FlexAWQSSZProcessor":
        from .flex_smooth import FlexAWQSSZProcessor
        return FlexAWQSSZProcessor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}") 
