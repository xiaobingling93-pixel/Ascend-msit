#  -*- coding: utf-8 -*-
#  Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""
Anti-outlier common utilities
"""

__all__ = [
    'VirtualVModuleFromQKVFused',
    'VirtualVModuleFromKVFused',
    'HookManager',
    'StatsCollector',
    'SubgraphRegistry',
    'IterSmoothContext',
    'IterSmoothConfig',
    'FlexSmoothQuantContext',
    'FlexSmoothQuantConfig',
    'FlexAWQSSZContext',
    'FlexAWQSSZConfig',
    'SmoothContext',
]

from .fused_linear import VirtualVModuleFromQKVFused, VirtualVModuleFromKVFused
from .smooth_components import (
    HookManager,
    StatsCollector,
    SubgraphRegistry
)
from .smooth_types import (
    IterSmoothContext,
    IterSmoothConfig,
    FlexSmoothQuantContext,
    FlexSmoothQuantConfig,
    FlexAWQSSZContext,
    FlexAWQSSZConfig,
    SmoothContext,
)

