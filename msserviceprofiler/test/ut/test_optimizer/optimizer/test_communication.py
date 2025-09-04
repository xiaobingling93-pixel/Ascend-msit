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

from pathlib import Path
from unittest.mock import MagicMock
import unittest

import pytest

from msserviceprofiler.modelevalstate.optimizer.communication import CommunicationForFile, CustomCommand


def test_init(tmpdir):
    # 创建临时文件路径
    work_dir = Path(tmpdir)
    cmd_file = work_dir.joinpath("parent", "cmd.txt")
    res_file = work_dir.joinpath("parent", "res.txt")

    # 测试当父目录不存在时，是否能正确创建
    assert not cmd_file.parent.exists()
    assert not res_file.parent.exists()
    comm = CommunicationForFile(cmd_file, res_file)
    assert cmd_file.parent.exists()
    assert res_file.parent.exists()
    assert comm.cmd_file_lock.exists()
    assert comm.res_file_lock.exists()

    # 测试当锁文件存在时，是否不会重新创建
    old_cmd_lock_mtime = comm.cmd_file_lock.stat().st_mtime
    old_res_lock_mtime = comm.res_file_lock.stat().st_mtime
    comm = CommunicationForFile(cmd_file, res_file)
    assert comm.cmd_file_lock.stat().st_mtime == old_cmd_lock_mtime
    assert comm.res_file_lock.stat().st_mtime == old_res_lock_mtime

    # 测试timeout参数是否正确设置
    comm = CommunicationForFile(cmd_file, res_file, timeout=200)
    assert comm.timeout == 200
