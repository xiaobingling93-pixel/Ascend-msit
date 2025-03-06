# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

from typing import List, Any, Dict
from transformers import PreTrainedModel

from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlierConfig
from msmodelslim.pytorch.llm_ptq.hooks.hook_def import ProcessHook


def get_norm_linear_subgraph(model: PreTrainedModel):
    norm_linear = {}
    layer_num = model.config.num_hidden_layers

    # 校验layer_num是否过大或过小
    if layer_num < 1 or layer_num > 999:
        raise ValueError(f"The number of hidden layers({layer_num}) is invalid. It must be between 1 and 999.")

    for layer in range(layer_num):
        input_layernorm = 'model.layers.' + str(layer) + '.input_layernorm'
        if layer % 2 != 0:
            q_proj = 'model.layers.' + str(layer) + '.self_attn.q_proj'
            norm_linear[input_layernorm] = [q_proj]
        else:
            q_proj = 'model.layers.' + str(layer) + '.self_attn.q_proj'
            k_proj = 'model.layers.' + str(layer) + '.self_attn.k_proj'
            v_proj = 'model.layers.' + str(layer) + '.self_attn.v_proj'
            norm_linear[input_layernorm] = [q_proj, k_proj, v_proj]

    return norm_linear


def modify_smooth_args(cfg: AntiOutlierConfig,
                       norm_name: str,
                       linear_names: str,
                       args: List[Any],
                       kwargs: Dict[str, Any]):
    # 针对该模型进行m4量化时，需要对特定层开启偏移
    if cfg.anti_method == 'm4':
        is_shift = False
        if 'norm' in norm_name:
            is_shift = True

        kwargs['is_shift'] = is_shift
        kwargs['alpha'] = cfg.alpha
    return args, kwargs


def get_hooks(model: PreTrainedModel):
    hooks = {
        ProcessHook.GET_NORM_LINEAR_SUBGRAPH: get_norm_linear_subgraph,
        ProcessHook.MODIFY_SMOOTH_ARGS: modify_smooth_args
    }
    return hooks
