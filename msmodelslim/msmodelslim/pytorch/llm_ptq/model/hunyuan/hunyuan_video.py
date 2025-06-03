# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

from typing import List, Any, Dict, Optional, Type, TYPE_CHECKING

import torch
import torch.nn as nn

from msmodelslim.pytorch.llm_ptq.model.base import ModelAdapter, ModelAdapterRegistry

if TYPE_CHECKING:
    from msmodelslim.pytorch.llm_ptq.anti_outlier.config import AntiOutlierConfig


@ModelAdapterRegistry.register("hunyuan_video")
class HunyuanVideoAdapter(ModelAdapter):
    def get_norm_linear_subgraph(self,
                                 cfg: 'AntiOutlierConfig',
                                 dummy_input: Optional[torch.Tensor] = None,
                                 norm_class: Optional[List[Type[nn.Module]]] = None):
        norm_linear = {}
        double_layer_num = self.model.config.mm_double_blocks_depth

        for layer in range(double_layer_num):
            input_layernorm = str(layer) + 'img_qkv_anti'
            img_qkv_proj = 'double_blocks.' + str(layer) + '.img_attn_qkv'

            norm_linear[input_layernorm] = [img_qkv_proj]

            input_layernorm = str(layer) + 'txt_qkv_anti'
            txt_qkv_proj = 'double_blocks.' + str(layer) + '.txt_attn_qkv'

            norm_linear[input_layernorm] = [txt_qkv_proj]

        return norm_linear

    def modify_smooth_args(self,
                           cfg: 'AntiOutlierConfig',
                           norm_name: str,
                           linear_names: str,
                           args: List[Any],
                           kwargs: Dict[str, Any]):
        # 针对该模型进行m4量化时，需要对特定层开启偏移
        if cfg.anti_method == 'm4':
            kwargs['is_shift'] = False
            kwargs['alpha'] = 0.5
        return args, kwargs
