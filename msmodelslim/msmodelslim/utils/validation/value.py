# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from msmodelslim.utils.exception import SchemaValidateError


def greater_than_zero(v: float) -> float:
    if v <= 0:
        raise SchemaValidateError("value must be greater than 0", 
                                  action="Please check the float value")
    return v
