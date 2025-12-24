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

"""
Pytest configuration for multimodal_vlm_v1 tests.

This conftest.py mocks the config initialization to avoid path validation errors
during import time.
"""

import sys
from unittest.mock import MagicMock

import pytest

from testing_utils.mock import mock_kia_library, mock_security_library, mock_init_config


def _mock_check_dirpath_before_read(path):
    """Mock function for check_dirpath_before_read that bypasses validation"""
    return path


# Mock necessary libraries before any imports that depend on them
# This must be executed at module level to ensure it runs before test imports
mock_init_config()
mock_kia_library()
mock_security_library()

_wcmatch_original = None
_wcmatch_mock_used = False

try:
    import wcmatch

    _wcmatch_original = sys.modules.get("wcmatch")
except ImportError:
    _wcmatch_mock_used = True
    _wcmatch_original = None
    sys.modules["wcmatch"] = MagicMock()
    sys.modules["wcmatch"].fnmatch = MagicMock()


def pytest_unconfigure(config):
    """在测试结束后清理按需 mock 的模块，恢复原始模块或属性（如果存在）。"""
    global _wcmatch_original, _wcmatch_mock_used

    # 1. 清理 wcmatch mock
    if _wcmatch_mock_used:
        if "wcmatch" in sys.modules:
            del sys.modules["wcmatch"]
        if _wcmatch_original is not None:
            sys.modules["wcmatch"] = _wcmatch_original
