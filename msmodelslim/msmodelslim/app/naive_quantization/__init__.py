# Copyright Huawei Technologies Co., Ltd. 2025. All rights reserved.

try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except ImportError:
    pass
