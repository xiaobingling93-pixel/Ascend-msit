# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import random
import numpy as np
import torch

import transformers

npu_available = False
try:
    import torch_npu
except ImportError:
    pass
else:
    npu_available = True


def process_item(item, device):
    if isinstance(item, torch.Tensor):
        return item.to(device)
    elif isinstance(item, (list, tuple)):
        return type(item)([process_item(x, device) for x in item])
    elif isinstance(item, dict):
        return {k: process_item(v, device) for k, v in item.items()}
    else:
        return item


def to_device(data, device):
    device = torch.device(device)
    
    if not isinstance(data, (list, tuple)):
        return process_item(data, device)
    
    prepared_data = []
    for batch in data:
        prepared_data.append(process_item(batch, device))
            
    return type(data)(prepared_data)
