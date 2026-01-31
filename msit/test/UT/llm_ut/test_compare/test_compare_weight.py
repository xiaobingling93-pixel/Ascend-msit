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
import os
import shutil
import stat
from glob import glob
from unittest import mock

import pytest
import pandas as pd
import torch
from safetensors.torch import save_file  

torch.manual_seed(0)


FILE_PERMISSION = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP
FAKE_GOLDEN_DATA_PATH = "./test_acc_cmp_fake_golden_data"
FAKE_MY_DATA_PATH = "./test_acc_cmp_fake_test_data"
OUTPUT_CSV_PATH = "./output_csv_path"


@pytest.fixture(scope='module', autouse=True)
def test_fake_golden_data_path():

    if not os.path.exists(FAKE_GOLDEN_DATA_PATH):
        os.makedirs(FAKE_GOLDEN_DATA_PATH, mode=0o750)
        tensors = {"weight":torch.randint(0, 1, (2, 2), dtype=torch.int8).to(dtype=torch.float32)}
        save_file(tensors, os.path.join(FAKE_GOLDEN_DATA_PATH, "model.safetensors"))
    yield FAKE_GOLDEN_DATA_PATH

    if os.path.exists(FAKE_GOLDEN_DATA_PATH):
        shutil.rmtree(FAKE_GOLDEN_DATA_PATH)


@pytest.fixture(scope='module', autouse=True)
def test_fake_quant_data_path():

    if not os.path.exists(FAKE_MY_DATA_PATH):
        os.makedirs(FAKE_MY_DATA_PATH, mode=0o750)
        tensors = {"weight":torch.randint(0, 1, (2, 2), dtype=torch.int8),
                   "weight_offset":torch.zeros(1),
                   "weight_scale":torch.ones(1)
                   }
        save_file(tensors, os.path.join(FAKE_MY_DATA_PATH, "model.safetensors"))
    yield FAKE_MY_DATA_PATH

    if os.path.exists(FAKE_MY_DATA_PATH):
        shutil.rmtree(FAKE_MY_DATA_PATH)


def test_compare_weight_given_loaded_data_when_valid_then_pass():
    from msit_llm.compare.cmp_weight import compare_weight
    if not os.path.exists(OUTPUT_CSV_PATH):
        os.makedirs(OUTPUT_CSV_PATH, mode=0o750)
    compare_weight(FAKE_GOLDEN_DATA_PATH, FAKE_MY_DATA_PATH, OUTPUT_CSV_PATH)
    csv_file_path = glob(f'{OUTPUT_CSV_PATH}/*.csv')[0]
    row_data = pd.read_csv(csv_file_path)
    assert row_data["cosine_similarity"][0] == 1.0

    if os.path.exists(OUTPUT_CSV_PATH):
        shutil.rmtree(OUTPUT_CSV_PATH)