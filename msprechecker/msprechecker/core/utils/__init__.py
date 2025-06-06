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

from msprechecker.core.utils.perm import FilePerm
from msprechecker.prechecker.register import show_check_result, CheckResult
from msprechecker.core.utils.macro_expander import MacroExpander
from msprechecker.core.utils.compiler import Compiler
from msprechecker.core.utils.result import Result, ResultStatus


def check_file_permission(filepath, domain="config", checker_name="file_perm"):
    try:
        perm = FilePerm(oct(os.stat(filepath).st_mode & 0o777))
        if perm > FilePerm(0o640):
            show_check_result(
                domain,
                checker_name,
                CheckResult.ERROR,
                action=f"请修改 {filepath} 权限为小于 640 (如 chmod 640 {filepath})",
                reason=f"当前权限为 {perm}，出于安全考虑，配置文件的权限应该不能超过 0o640",
            )
            return False
    except Exception as e:
        show_check_result(
            domain,
            checker_name,
            CheckResult.ERROR,
            action=f"无法检查 {filepath} 权限",
            reason=str(e),
        )
        return False
    return True
