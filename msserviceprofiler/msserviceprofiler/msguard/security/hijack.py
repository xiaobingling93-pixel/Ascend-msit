# -*- coding: utf-8 -*-
# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from ..validation import validate_params
from ..constraints import Rule


@validate_params(
    {"path": Rule.input_file_exec}
)
def update_env_s(env_var: str, path: str, prepend: bool = True) -> None:
    if not isinstance(env_var, str):
        raise TypeError("Environment variable name must be str")
    
    if not os.path.isabs(path):
        raise ValueError(f"Relative paths are not allowed: {path}")

    current_value = os.environ.get(env_var)
    
    if current_value:
        parts = [p for p in current_value.split(os.pathsep)]
        if prepend:
            parts.insert(0, path)
        else:
            parts.append(path)
        new_value = os.pathsep.join(parts)
    else:
        new_value = path
    
    os.environ[env_var] = new_value
