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
