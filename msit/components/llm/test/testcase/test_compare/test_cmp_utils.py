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
import pytest
import torch

from msit_llm.compare.cmp_utils import BasicDataInfo


@pytest.fixture(scope='module', autouse=True)
def golden_data_path():
    golden_data_path = "msit_dump_20240101_000000/torch_tensors/npu0_11111/1"
    yield golden_data_path


@pytest.fixture(scope='module', autouse=True)
def my_data_path():
    my_data_path = "msit_dump_20240101_000000/tensors/0_22222/2"
    yield my_data_path


@pytest.fixture(scope='module', autouse=True)
def my_data_path():
    sub_path = "3_Prefill_layer/0_Attention/3_SelfAttention"
    yield sub_path


@pytest.fixture(scope='module', autouse=True)
def unvalid_data_path1():
    unvalid_data_path1 = "tensor_dump/tensors/0_22222/2"
    yield unvalid_data_path1


@pytest.fixture(scope='module', autouse=True)
def unvalid_data_path2():
    unvalid_data_path2 = "msit_dump_20240101_000000/tensors/0_22222/not_int"
    yield unvalid_data_path2


def test_get_token_id_from_golden_data_path(golden_data_path, my_data_path):
    data_info = BasicDataInfo(my_data_path, golden_data_path)
    assert data_info.token_id == 1


def test_get_token_id_from_my_data_path(golden_data_path, my_data_path):
    data_info = BasicDataInfo(golden_data_path, my_data_path)
    assert data_info.token_id == 2


def test_get_token_id_from_golden_data_path_sub(golden_data_path, my_data_path, sub_path):
    my_data_path = os.path.join(my_data_path, sub_path)
    golden_data_path = os.path.join(golden_data_path, sub_path)
    data_info = BasicDataInfo(my_data_path, golden_data_path)
    assert data_info.token_id == 1


def test_get_token_id_from_my_data_path_sub(golden_data_path, my_data_path, sub_path):
    my_data_path = os.path.join(my_data_path, sub_path)
    golden_data_path = os.path.join(golden_data_path, sub_path)
    data_info = BasicDataInfo(golden_data_path, my_data_path)
    assert data_info.token_id == 2


def test_get_token_id_unvalid1(golden_data_path, unvalid_data_path1):
    data_info = BasicDataInfo(golden_data_path, unvalid_data_path1)
    assert data_info.token_id == 0


def test_get_token_id_unvalid2(golden_data_path, unvalid_data_path2):
    data_info = BasicDataInfo(golden_data_path, unvalid_data_path2)
    assert data_info.token_id == 0


def test_given_token_id_data_id(golden_data_path, my_data_path):
    data_info = BasicDataInfo(golden_data_path, my_data_path, token_id=3, data_id=4)
    assert data_info.token_id == 3
    assert data_info.data_id == 4


def test_given_data_id(golden_data_path, my_data_path):
    data_info = BasicDataInfo(golden_data_path, my_data_path, data_id=4)
    assert data_info.token_id == 2
    assert data_info.data_id == 4


def test_given_token_id(golden_data_path, my_data_path):
    data_info1 = BasicDataInfo(golden_data_path, my_data_path, token_id=3)
    assert data_info1.token_id == 3
    assert data_info1.data_id == 0

    data_info2 = BasicDataInfo(golden_data_path, my_data_path, token_id=4)
    assert data_info2.token_id == 4
    assert data_info2.data_id == 1