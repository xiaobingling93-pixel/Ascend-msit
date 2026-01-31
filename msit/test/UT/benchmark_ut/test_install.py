# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------
import sys
from collections import namedtuple
from unittest.mock import patch, MagicMock
import pytest

ACL_RUNTIME = namedtuple("aclruntime", "key")("aclruntime")
AIS_BENCH_RUNTIME = namedtuple("ais_bench", "key")("ais-bench")


def test_install_check_given_all_installed_then_pass():
    mock_pkg = MagicMock()
    mock_pkg.working_set = [ACL_RUNTIME, AIS_BENCH_RUNTIME]
    with patch.dict(sys.modules, {"pkg_resources": mock_pkg}):
        from msit_benchmark.__install__ import BenchmarkInstall
        assert BenchmarkInstall().check() == "OK"


@pytest.mark.filterwarnings("ignore:Deprecated call to:DeprecationWarning")
@pytest.mark.filterwarnings("ignore:pkg_resources is deprecated:DeprecationWarning")
@patch("subprocess.run")
def test_install_build_extra_given_valid_then_pass(mock_run):
    from msit_benchmark.__install__ import BenchmarkInstall
    BenchmarkInstall().build_extra()
    mock_run.assert_called_once()


@patch("subprocess.run")
def test_install_download_extra_given_valid_then_pass(mock_run):
    from msit_benchmark.__install__ import BenchmarkInstall
    BenchmarkInstall().download_extra(dest=".")
    mock_run.assert_called_once()
