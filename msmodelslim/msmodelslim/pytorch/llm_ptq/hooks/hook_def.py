# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.

from enum import Enum


class ProcessHook(Enum):
    GET_NORM_LINEAR_SUBGRAPH = 0
    MODIFY_SMOOTH_ARGS = 1
    MODIFY_QUANTIZER_ARGS = 2
