#  -*- coding: utf-8 -*-
#  Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
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
Common utilities for anti-outlier implementation modules
"""

__all__ = [
    # Alpha-Beta Search
    'FlexSmoothAlphaBetaSearcher',
    'FlexAWQSSZAlphaBetaSearcher',
    'BaseAlphaBetaSearcher',
    'quant_int8sym',
    'quant_int8asym',
    # Scale Computation
    'IterSmoothScaleCalculator',
    'FlexSmoothScaleCalculator',
    'FlexAWQSSZScaleCalculator',
    'BaseScaleCalculator',
    'validate_and_process_tensors',
    'compute_weight_scale',
    'compute_multi_weight_scale',
    'apply_smooth_scale_shift',
    'prepare_mqga_parameters',
    'reduce_scales_for_mqga_mean',
    'reduce_scales_for_mqga_max',
    'MQGAScaleParams',
    # Subgraph Fusion
    'SubgraphFusionFactory',
    'SubgraphFusionStrategy',
    'OVSubgraphFusion',
    'UpDownSubgraphFusion',
    'LinearLinearSubgraphFusion',
    'NormLinearSubgraphFusion',
]

from .alpha_beta_search import (
    BaseAlphaBetaSearcher,
    FlexAWQSSZAlphaBetaSearcher,
    FlexSmoothAlphaBetaSearcher,
    quant_int8asym,
    quant_int8sym
)

from .scale_computation import (
    BaseScaleCalculator,
    FlexAWQSSZScaleCalculator,
    FlexSmoothScaleCalculator,
    IterSmoothScaleCalculator,
    MQGAScaleParams,
    apply_smooth_scale_shift,
    compute_multi_weight_scale,
    compute_weight_scale,
    prepare_mqga_parameters,
    reduce_scales_for_mqga_max,
    reduce_scales_for_mqga_mean,
    validate_and_process_tensors
)

from .subgraph_fusion import (
    LinearLinearSubgraphFusion,
    NormLinearSubgraphFusion,
    OVSubgraphFusion,
    SubgraphFusionFactory,
    SubgraphFusionStrategy,
    UpDownSubgraphFusion
)

