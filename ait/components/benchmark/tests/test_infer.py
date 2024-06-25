# Copyright (c) 2023-2023 Huawei Technologies Co., Ltd.
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

import pytest

from msit_benchmark.__main__ import get_cmd_instance
try:
    import acl
except ModuleNotFoundError:
    acl = None

@pytest.fixture(scope='module', autouse=True)
def build_extra():
    from msit_benchmark.__install__ import BenchmarkInstall

    BenchmarkInstall().build_extra()

@pytest.fixture(scope='module', autouse=True)
def build_test_model():
    import torch

    mm = torch.Linear(32, 32)
    torch.onnx.export(mm, torch.ones([1, 32]), 'foo.onnx')


def benchmark_argparse(argv):
    aa = get_cmd_instance()
    parser = argparse.ArgumentParser()
    aa.add_arguments(parser)
    args_parser = parser.parse_args(argv)
    return args_parser

@pytest.mark.skipif(acl is None, reason="missing CANN tolkit")
def test_install_build_extra_given_valid_then_pass():
    from msit_benchmark.__install__ import BenchmarkInstall

    BenchmarkInstall().build_extra()

    aa = get_cmd_instance()
    args = benchmark_argparse(["--model"])
    aa.handle(args)


