# Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

__all__ = [
    'BaseConfig',
    'WeightActivationQuantConfig',
    'WeightQuantConfig',
    'SparseQuantConfig',
    'KVQuantConfig',
]


from .base_config import BaseConfig
from .weight_activation_quant_config import WeightActivationQuantConfig
from .weight_quant_config import WeightQuantConfig
from .sparse_quant_config import SparseQuantConfig
from .kv_quant_config import KVQuantConfig