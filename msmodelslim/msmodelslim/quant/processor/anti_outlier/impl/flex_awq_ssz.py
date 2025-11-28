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


import torch
from torch import nn
from msmodelslim.core.QAL.qregistry import QFuncRegistry
from msmodelslim.core.QAL.qtypes import (
    Subgraph,
    NormLinearSubgraph,
    LinearLinearSubgraph,
    OVSubgraph,
    UpDownSubgraph,
)
from msmodelslim.quant.processor.anti_outlier.common import (
    SmoothContext,
    FlexAWQSSZConfig,
)

from msmodelslim.utils.logging import get_logger

from .common.scale_computation import (
    validate_and_process_tensors,
    compute_weight_scale,
    compute_multi_weight_scale,
    FlexAWQSSZScaleCalculator
)

from .common.alpha_beta_search import FlexAWQSSZAlphaBetaSearcher
from .common.subgraph_fusion import SubgraphFusionFactory


@torch.no_grad()
def get_optimal_alpha_beta(config: FlexAWQSSZConfig, act, linear: nn.Linear):
    if config.alpha is None:
        searcher = FlexAWQSSZAlphaBetaSearcher(qconfig=config.qconfig)
        best_alpha, normal_mse_best = searcher.search_alpha(act, linear)
        get_logger().debug(f"Found optimal alpha: {best_alpha:.6f}, MSE: {normal_mse_best:.6f}")
    else:
        best_alpha = config.alpha
        get_logger().debug(f"Using provided alpha: {best_alpha}")
    best_beta = 0
    return best_alpha, best_beta


@torch.no_grad()
@QFuncRegistry.register(dispatch_key=(OVSubgraph, 1), api_name="flex_awq_ssz")
def flex_awq_ssz_impl_ov(
    subgraph: Subgraph, config: FlexAWQSSZConfig, context: SmoothContext
) -> None:   
    tmp_device = next(subgraph.v_proj.parameters()).device
    dtype = subgraph.o_proj.weight.dtype
    act = validate_and_process_tensors(context, tmp_device, dtype)
    # flex awq ssz 的 act 尺度计算使用mean
    act_scales = torch.mean(torch.abs(act), dim=0, keepdim=True)[0]
    best_alpha, best_beta = get_optimal_alpha_beta(config, act, subgraph.o_proj)
    
    w_scale = compute_weight_scale(subgraph.o_proj.weight, dtype)
    calculator = FlexAWQSSZScaleCalculator(alpha=best_alpha, beta=best_beta)
    o_scales, v_scales = calculator.compute_ov_scales(
        act_scales, w_scale,
        subgraph.num_attention_heads,
        subgraph.key_value_heads
    )
    
    SubgraphFusionFactory.apply_fusion_to_subgraph(
        subgraph,
        scales={'o_scales': o_scales, 'v_scales': v_scales}
    )
    get_logger().debug("OV smoothing completed successfully")
    return


@torch.no_grad()
@QFuncRegistry.register(dispatch_key=(UpDownSubgraph, 1), api_name="flex_awq_ssz")
def flex_awq_ssz_impl_up_down(
    subgraph: Subgraph, config: FlexAWQSSZConfig, context: SmoothContext
) -> None:
    tmp_device = next(subgraph.up_proj.parameters()).device
    dtype = subgraph.down_proj.weight.dtype
    act = validate_and_process_tensors(context, tmp_device, dtype)
    act_scales = torch.mean(torch.abs(act), dim=0, keepdim=True)[0]
    best_alpha, best_beta = get_optimal_alpha_beta(config, act, subgraph.down_proj)
    w_scale = compute_weight_scale(subgraph.down_proj.weight, dtype)
    calculator = FlexAWQSSZScaleCalculator(alpha=best_alpha, beta=best_beta)
    scales = calculator.compute_smooth_scale(act_scales, w_scale)
    SubgraphFusionFactory.apply_fusion_to_subgraph(
        subgraph,
        scales={'scales': scales}
    )
    get_logger().debug("Up-Down smoothing completed successfully")
    return


@torch.no_grad()
@QFuncRegistry.register(dispatch_key=(LinearLinearSubgraph, 1), api_name="flex_awq_ssz")
def flex_awq_ssz_impl_linear_linear(
    subgraph: Subgraph, config: FlexAWQSSZConfig, context: SmoothContext
) -> None:
    tmp_device = next(subgraph.linear1.parameters()).device
    dtype = subgraph.linear2.weight.dtype
    act = validate_and_process_tensors(context, tmp_device, dtype)
    act_scales = torch.mean(torch.abs(act), dim=0, keepdim=True)[0]
    best_alpha, best_beta = get_optimal_alpha_beta(config, act, subgraph.linear2)
    w_scale = compute_weight_scale(subgraph.linear2.weight, dtype)
    calculator = FlexAWQSSZScaleCalculator(alpha=best_alpha, beta=best_beta)
    scales = calculator.compute_smooth_scale(act_scales, w_scale)
    SubgraphFusionFactory.apply_fusion_to_subgraph(
        subgraph,
        scales={'scales': scales}
    )
    get_logger().debug("Linear-Linear smoothing completed successfully")
    return


@torch.no_grad()
@QFuncRegistry.register(dispatch_key=(NormLinearSubgraph, 1), api_name="flex_awq_ssz")
def flex_awq_ssz_impl_norm_linear(
    subgraph: Subgraph, config: FlexAWQSSZConfig, context: SmoothContext
) -> None:
    tmp_device = next(subgraph.norm.parameters()).device
    dtype = subgraph.linears[0].weight.dtype
    act = validate_and_process_tensors(context, tmp_device, dtype)
    act_scales = torch.mean(torch.abs(act), dim=0, keepdim=True)[0]
    # 仅用前两层做校准
    if len(subgraph.linears) > 3:
        fc_weights = torch.cat([fc.weight for fc in subgraph.linears[0:2]], dim=0)
    else:
        fc_weights = torch.cat([fc.weight for fc in subgraph.linears], dim=0)
    
    get_logger().debug(
        "Activation scale shape: %s, Weight shape: %s",
        act_scales.shape, fc_weights.shape
    )
    merged_linear = nn.Linear(
        in_features=fc_weights.shape[1],
        out_features=fc_weights.shape[0],
        bias=False,
        device=fc_weights.device,
        dtype=fc_weights.dtype
    )
    merged_linear.weight.data = fc_weights
    best_alpha, best_beta = get_optimal_alpha_beta(config, act, merged_linear)

    weights = [fc.weight for fc in subgraph.linears]
    w_scale = compute_multi_weight_scale(weights, dtype)
    calculator = FlexAWQSSZScaleCalculator(alpha=best_alpha, beta=best_beta)
    scales = calculator.compute_smooth_scale(act_scales, w_scale)
    get_logger().debug("Computed smooth scales shape: %s", scales.shape)
    SubgraphFusionFactory.apply_fusion_to_subgraph(
        subgraph,
        scales={'scales': scales}
    )
    get_logger().debug("Norm-Linear smoothing completed successfully")
    return