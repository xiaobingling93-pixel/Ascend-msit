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

import re
import shlex
import shutil
import subprocess

from .exception import CSVInjectionError


CSV_INJECTION_PATTERN = re.compile(r'^[＋－＝％＠\+\-=%@]|;[＋－＝％＠\+\-=%@]')


# csv injection
def is_safe_csv_value(value: str) -> bool:
    if not isinstance(value, str):
        return True
    
    try:
        float(value)
    except ValueError:
        return not bool(CSV_INJECTION_PATTERN.search(value))

    return True


def sanitize_csv_value(value: str, errors: str = 'strict') -> str:
    if errors == 'ignore' or is_safe_csv_value(value):
        return value
    elif errors == 'replace':
        return "'" + value
    else:
        err_msg = f'Malicious value is not allowed to be written into the csv: {value}'
        raise CSVInjectionError(err_msg)


# command injection
def sanitize_cmd(cmd) -> str:
    if not cmd:
        raise ValueError(f"Invalid command: {cmd!r}")
    
    if isinstance(cmd, str):
        cmd = shlex.split(cmd)
    elif not isinstance(cmd, list):
        raise TypeError
    
    if not shutil.which(cmd[0]):
        raise FileNotFoundError
    
    return cmd


def run_s(cmd, **kwargs) -> subprocess.CompletedProcess:
    cmd = sanitize_cmd(cmd)
    return subprocess.run(cmd, shell=False, **kwargs)


def popen_s(cmd, **kwargs) -> subprocess.CompletedProcess:
    cmd = sanitize_cmd(cmd)
    return subprocess.Popen(cmd, shell=False, **kwargs)


def checkoutput_s(cmd, **kwargs):
    cmd = sanitize_cmd(cmd)
    return subprocess.check_output(cmd, shell=False, **kwargs)
