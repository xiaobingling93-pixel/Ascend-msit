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

import json
from typing import Type, Dict, TypeVar, ClassVar, Any, Generic, Optional, Union

from pydantic import BaseModel, ConfigDict
from pydantic_core import PydanticCustomError

from msmodelslim.utils.logging import get_logger

T = TypeVar('T', bound='BaseModel')


class BaseAutoConfig(BaseModel, Generic[T]):
    """工厂模式混入类，支持基于type字段的自动子类选择"""
    _is_factory_validating: ClassVar[bool] = False
    model_config = ConfigDict(extra="forbid")

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs):
        if 'type' not in cls.model_fields:
            raise TypeError(f"Must provide a type field for {cls.__bases__}'s subclass")

        if BaseAutoConfig in cls.__bases__:
            cls._registry: Dict[str, Type[T]] = {}
            get_logger().debug(f"[Utils] Create registry for {cls.__name__}")
            return super().__pydantic_init_subclass__(**kwargs)

        cls._registry[cls.model_fields['type'].default] = cls
        return super().__pydantic_init_subclass__(**kwargs)

    @classmethod
    def model_validate(
            cls,
            obj: Any,
            *,
            strict: Optional[bool] = None,
            from_attributes: Optional[bool] = None,
            context: Optional[Any] = None,
    ) -> T:
        if isinstance(obj, dict) and 'type' in obj and not cls._is_factory_validating:
            return cls._validate_dict_with_type(obj)
        return super().model_validate(obj)

    @classmethod
    def model_validate_json(
            cls,
            json_data: Union[str, bytes, bytearray],
            *,
            strict: Optional[bool] = None,
            context: Optional[Any] = None,
    ) -> T:
        return cls.model_validate(json.loads(json_data))

    @classmethod
    def _validate_dict_with_type(cls: Type[T], obj: dict) -> T:
        if 'type' not in obj:
            raise PydanticCustomError(
                "invalid_auto_config_type",
                "missing type field in input dict",
                None,
            )
        type_value = obj.get('type')
        if type_value not in cls._registry:
            raise PydanticCustomError(
                "invalid_auto_config_type",
                "invalid auto config type: {type_value}, available types: {available_types}",
                {"type_value": type_value, "available_types": list(cls._registry.keys())},
            )
        cls._is_factory_validating = True
        try:
            return cls._registry[type_value].model_validate(obj)
        finally:
            cls._is_factory_validating = False
