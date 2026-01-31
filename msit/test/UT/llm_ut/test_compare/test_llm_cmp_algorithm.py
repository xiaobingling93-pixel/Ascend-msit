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
import os.path

import pytest
import torch

from components.utils.cmp_algorithm import cosine_similarity, max_relative_error, mean_relative_error, \
    relative_euclidean_distance, l1_norm


@pytest.fixture(scope='module', autouse=True)
def golden_data():
    golden_data = torch.ones((2, 3))
    yield golden_data


@pytest.fixture(scope='module', autouse=True)
def test_data():
    test_data = torch.ones((2, 3))
    yield test_data


def test_cosine_similarity(golden_data, test_data):
    res, message = cosine_similarity(golden_data.reshape(-1), test_data.reshape(-1))
    assert res == 1.0
    assert message == ''


def test_max_relative_error(golden_data, test_data):
    res, message = max_relative_error(golden_data, test_data)
    assert res == 0.0
    assert message == ''


def test_mean_relative_error(golden_data, test_data):
    res, message = mean_relative_error(golden_data, test_data)
    assert res == 0.0
    assert message == ''


def test_relative_euclidean_distance(golden_data, test_data):
    res, message = relative_euclidean_distance(golden_data, test_data)
    assert res == 0.0
    assert message == ''


def test_relative_euclidean_distance_when_low_acc(golden_data, test_data):
    test_data = 10 * test_data
    res, message = relative_euclidean_distance(golden_data, test_data)
    assert res == 9.0
    assert message == ''

def test_l1_norm(golden_data, test_data):
    res, message = l1_norm(golden_data, test_data)
    assert res == 0.0
    assert message == ''