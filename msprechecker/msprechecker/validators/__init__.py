# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2025-2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          `http://license.coscl.org.cn/MulanPSL2`
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

__all__ = ["get_validator"]


_validator_registry = {}


def register_validator(validator_type: str, validator_cls: type):
    _validator_registry[validator_type] = validator_cls


def _lazy_register_validators():
    from .range import RangeValidator
    from .cmp import (
        GreaterThanValidator, LessThanValidator, EqualValidator,
        GEValidator, LEValidator, NEValidator
    )
    from .enum import EnumValidator
    from .path import PathValidator
    from .docker import DockerValidator

    aliases = {
        "range": RangeValidator,
        "enum": EnumValidator, "in": EnumValidator,
        "eq": EqualValidator, "==": EqualValidator,
        "lt": LessThanValidator, "<": LessThanValidator,
        "gt": GreaterThanValidator, ">": GreaterThanValidator,
        "ge": GEValidator, ">=": GEValidator,
        "le": LEValidator, "<=": LEValidator,
        "ne": NEValidator, "!=": NEValidator,
        "path": PathValidator,
        "docker": DockerValidator
    }

    for key, cls in aliases.items():
        register_validator(key, cls)


def get_validator(validator_type: str) -> type:
    if not _validator_registry:
        _lazy_register_validators()

    if validator_type not in _validator_registry:
        raise ValueError(f"Unknown validator type: {validator_type}")
        
    return _validator_registry[validator_type]
