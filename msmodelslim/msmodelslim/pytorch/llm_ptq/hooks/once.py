#  Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
from functools import wraps

import torch


def once(func):
    is_used = False

    @wraps(func)
    def wrapper(*args, **kwargs):
        nonlocal is_used
        if is_used:
            return

        func(*args, **kwargs)
        is_used = True
        return

    return wrapper


@once
def register_bias_func(model: torch.nn.Module):
    for _, module in model.named_modules():
        module.has_origin_bias = (isinstance(module, torch.nn.Linear)
                                  and hasattr(module, 'bias') and module.bias is not None)


register_bias = register_bias_func
