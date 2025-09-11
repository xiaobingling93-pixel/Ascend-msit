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
from abc import abstractmethod

import torch
from torch import nn

from msmodelslim.utils.exception import SpecError, ToDoError
from msmodelslim.utils.logging import get_logger


class VirtualVModuleBase(nn.Module):
    @abstractmethod
    def update_weights(self):
        raise ToDoError("update_weights is not implemented.",
                        action="Please implement this method in the subclass.")

    def forward(self, x):
        raise SpecError("VirtualVModuleBase is not supported for forward.",
                        action="Please do Not use this module for forward.")


class VirtualVModuleFromQKVFused(VirtualVModuleBase):
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

    def update_weights(self):
        """将更新后的 V 权重更新回原始的 QKV 模块"""
        qkv_weight = self.qkv_module.weight
        qkv_bias = getattr(self.qkv_module, 'bias', None)

        # 计算每个头的维度
        head_dim = qkv_weight.shape[1] // self.num_attention_heads

        # 根据注意力类型计算 V 部分的起始和结束索引
        v_start, v_end = self._get_v_indices(head_dim)

        # 更新 V 部分的权重
        with torch.no_grad():
            qkv_weight[v_start:v_end] = self.weight.data

            # 处理bias更新：如果smooth后产生了bias，需要更新到qkv_module
            if hasattr(self, 'bias') and self.bias is not None:
                if qkv_bias is not None:
                    # 原始qkv_module有bias，直接更新V部分
                    qkv_bias[v_start:v_end] = self.bias.data
                else:
                    # 原始qkv_module没有bias，但smooth后产生了bias，需要创建新的bias
                    # 参考iter_smooth.py的模式：创建全零bias，然后更新V部分
                    device = qkv_weight.device
                    dtype = qkv_weight.dtype
                    new_bias = torch.zeros(qkv_weight.shape[0], device=device, dtype=dtype)
                    new_bias[v_start:v_end] = self.bias.data
                    self.qkv_module.bias = nn.Parameter(new_bias)

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
        self.weight = nn.Parameter(qkv_weight[v_start:v_end].clone())

        # 提取 V 部分的偏置
        if qkv_bias is not None:
            self.bias = nn.Parameter(qkv_bias[v_start:v_end].clone())
        else:
            self.bias = None


class VirtualVModuleFromKVFused(VirtualVModuleBase):
    """虚拟 V 模块，用于处理 DeepSeek V3 的 KV 融合（按 head 展开，行维度按 [K_nope, V] 拼接）。

    假设 qk_nope_head_dim 与 v_head_dim 相等，并从权重的 out_features 自动推导每个 head 的尺寸。
    """

    def __init__(self, kv_module: nn.Linear,
                 num_attention_heads: int,
                 qk_nope_head_dim: int,
                 v_head_dim: int):
        super().__init__()
        self.kv_module = kv_module
        self.num_attention_heads = num_attention_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.v_head_dim = v_head_dim

        # 从权重维度推导每个 head 的行数，并强制 K_nope 与 V 等长
        out_features = self.kv_module.weight.data.size(0)
        per_head_out = out_features // self.num_attention_heads
        if per_head_out != self.qk_nope_head_dim + self.v_head_dim:
            raise SpecError(f'KV-fused module weight dimension mismatch, please check.',
                            action='Please ensure the weight out_features dimension '
                                   'is (qk_nope_head_dim + v_head_dim) * num_attention_heads.')

        # 计算所有 V 行的索引：对每个 head，选择后半段的 v_head_dim 行
        kv_w = self.kv_module.weight.data.view(self.num_attention_heads, per_head_out, -1)
        v_w = kv_w[:, self.qk_nope_head_dim:, :].contiguous().view(-1, kv_w.size(-1))
        self.weight = nn.Parameter(v_w.clone())
        if self.kv_module.bias is not None:
            kv_b = self.kv_module.bias.data.view(self.num_attention_heads, per_head_out)
            v_b = kv_b[:, self.qk_nope_head_dim:].contiguous().view(-1)
            self.bias = nn.Parameter(v_b.clone())

    def update_weights(self):
        """将更新后的 V 权重更新回原始的 QKV 模块"""
        with torch.no_grad():
            out_features = self.kv_module.weight.data.size(0)
            per_head_out = out_features // self.num_attention_heads
            kv_w = self.kv_module.weight.data.view(self.num_attention_heads, per_head_out, -1)
            kv_w[:, self.qk_nope_head_dim:, :] = self.weight.data.view(self.num_attention_heads, self.v_head_dim, -1)

            # 处理bias更新：如果smooth后产生了bias，需要更新到kv_module
            if hasattr(self, 'bias') and self.bias is not None:
                if self.kv_module.bias is not None:
                    # 原始kv_module有bias，直接更新V部分
                    kv_b = self.kv_module.bias.data.view(self.num_attention_heads, per_head_out)
                    # self.bias是一维的，需要reshape为[num_attention_heads, v_head_dim]
                    v_bias_reshaped = self.bias.data.view(self.num_attention_heads, self.v_head_dim)
                    kv_b[:, self.qk_nope_head_dim:] = v_bias_reshaped
                else:
                    # 原始kv_module没有bias，但smooth后产生了bias，需要创建新的bias
                    # 创建全零的bias，然后更新V部分
                    new_bias = torch.zeros(out_features, device=self.kv_module.weight.device,
                                           dtype=self.kv_module.weight.dtype)
                    kv_b = new_bias.view(self.num_attention_heads, per_head_out)
                    # self.bias是一维的，需要reshape为[num_attention_heads, v_head_dim]
                    v_bias_reshaped = self.bias.data.view(self.num_attention_heads, self.v_head_dim)
                    kv_b[:, self.qk_nope_head_dim:] = v_bias_reshaped
                    self.kv_module.bias = nn.Parameter(new_bias)
