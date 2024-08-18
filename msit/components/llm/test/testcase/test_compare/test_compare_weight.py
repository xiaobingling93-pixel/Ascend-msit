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
import shutil
import stat
import safetensor
from safetensors.torch import save_file

import pytest
from unittest import mock
import torch
import numpy as np

import msit_llm.compare



FILE_PERMISSION = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP
FAKE_GOLDEN_DATA_PATH = "test_acc_cmp_fake_golden_data"
FAKE_MY_DATA_PATH = "test_acc_cmp_fake_test_data"


@pytest.fixture(scope='module')
def test_fake_golden_data_path():

    if not os.path.exists(FAKE_GOLDEN_DATA_PATH):
        os.makedirs(os.path.join(FAKE_GOLDEN_DATA_PATH), mode=0o750)
        tensors = {"weight":torch.random((2, 2))}
        save_file(tensors, os.path.join(FAKE_GOLDEN_DATA_PATH, "model.safetensors"))
    yield FAKE_GOLDEN_DATA_PATH

    if os.path.exists(FAKE_GOLDEN_DATA_PATH):
        shutil.rmtree(FAKE_GOLDEN_DATA_PATH)  

@pytest.fixture(scope='module')
def test_fake_golden_data_path():

    if not os.path.exists(FAKE_GOLDEN_DATA_PATH):
        os.makedirs(os.path.join(FAKE_GOLDEN_DATA_PATH), mode=0o750)
        tensors = {"weight":torch.random((2, 2)),"weight_offset":torch(0),"weight_scale":torch(1)}
        save_file(tensors, os.path.join(FAKE_GOLDEN_DATA_PATH, "model.safetensors"))
    yield FAKE_GOLDEN_DATA_PATH

    if os.path.exists(FAKE_GOLDEN_DATA_PATH):
        shutil.rmtree(FAKE_GOLDEN_DATA_PATH)  

def test_compare_weight_given_loaded_data_when_valid_then_pass():
    msit_llm.compare.cmp_weight.compare_weight(FAKE_GOLDEN_DATA_PATH, FAKE_MY_DATA_PATH, "./")
    row_data = msit_llm.compare.read_data("./")
    assert row_data["cosine_similarity"] == 1.0