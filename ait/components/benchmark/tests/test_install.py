# Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
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
import sys
import argparse
from collections import namedtuple


import pytest

ACL_RUNTIME = namedtuple("aclruntime", "key")("aclruntime")
AIS_BENCH_RUNTIME = namedtuple("ais_bench", "key")("ais-bench")


def test_install_check_given_all_installed_then_pass():
    sys.modules["pkg_resources"] = namedtuple("pkg_resources", "working_set")([ACL_RUNTIME, AIS_BENCH_RUNTIME])
    from msit_benchmark.__install__ import BenchmarkInstall

    assert BenchmarkInstall().check() == "OK"


def test_install_build_extra_given_valid_then_pass():
    from msit_benchmark.__install__ import BenchmarkInstall

    BenchmarkInstall().build_extra()


def test_install_download_extra_given_valid_then_pass():
    from msit_benchmark.__install__ import BenchmarkInstall

    BenchmarkInstall().download_extra(dest=".")
