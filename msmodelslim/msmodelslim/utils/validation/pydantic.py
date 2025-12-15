# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
Pydantic 特定的验证函数包装器。

这些函数专门为 Pydantic AfterValidator 设计，能够自动从 ValidationInfo 获取字段名。
基于 value.py 中的通用验证函数进行包装。
"""

from typing import Any, Union

from pydantic import ValidationInfo

from msmodelslim.utils.validation.value import (
    greater_than_zero as _greater_than_zero,
)


def greater_than_zero(v: Any, info: Union[ValidationInfo, None] = None) -> Any:
    """
    给 Pydantic AfterValidator 使用的大于零校验包装。
    自动从 ValidationInfo 获取字段名，然后调用通用的验证函数。
    
    Args:
        v: 要验证的数值
        info: Pydantic ValidationInfo 对象，用于获取字段名（可选）
        
    Returns:
        验证后的数值
        
    Example:
        ```python
        from pydantic import BaseModel, Field, AfterValidator
        from typing import Annotated
        
        class Config(BaseModel):
            timeout: Annotated[int, AfterValidator(greater_than_zero)] = 60
        ```
    """
    field_name = "value"
    if info is not None:
        # 在 Pydantic v2 中，ValidationInfo 有 field_name 属性
        field_name = getattr(info, 'field_name', field_name) or field_name
    return _greater_than_zero(v, param_name=field_name)
