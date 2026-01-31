# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------
import re


def is_word_embedding(name):
    return ('embed_tokens' in name or 'word_embeddings' in name) \
        and 'word_embeddings_layernorm' not in name


ATB_TORCH_BUILT_IN_OP_OUTPUT_MAPPING = {
    "LayerNormOperation": "LayerNorm",
    "LinearOperation": "Linear"
}


ATB_TORCH_CUSTOM_OP_OUTPUT_MAPPING = {
    "CommonLayer_outtensor0": ["GLMBlock_output_0", "BloomBlock_output_0"],
    "MlpGateLayerV2": ["BloomMLP", "MLP"],
    "RmsNormOperation": ["RMSNorm"],
    "SelfAttentionOperation": ["CoreAttention"],
}


ATB_QUANT_FLOAT_NODE_MAPPING = {
    "LinearQuantOperation": "LinearOperation",
    "LinearDequantOnly": "LinearNoQuant",
    "LinearRowParallelNoAdd": "LinearRowParallelAndAdd",
}


LAYER_OP_MAPPING_DICT = {
    'wordembedding': is_word_embedding,
    re.compile(r"^RmsNormOperation_\d+$"): 'root.model.norm',
    'lmhead': 'lm_head',
    'layernormoperation_35': 'embeddings_layernorm',
    'layernormoperation_66': 'ln_f'
}


QWEN_OP_MAPPING = {
        'qkv': ['self_attn.v_proj', 'self_attn.k_proj', 'self_attn.q_proj']
}


MODULE_MAPPING_DICT = {
    # ATB算子: [Pytorch算子列表]
    'Attention': ['self_attn', 'self_attention', 'Attention'],
    'Mlp': ['mlp', 'Mlp'],
    'mlp': ['mlp'],
    'self_attn': ['self_attn'],
    'self_attention': ['self_attention']
}