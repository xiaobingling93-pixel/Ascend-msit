# Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from test.resources.sample_net_torch import LrdSampleNetwork
import pytest
import numpy as np
import torch

from ascend_utils.common.utils import count_parameters
from msmodelslim.pytorch.low_rank_decompose import low_rank_decompose


@pytest.fixture(scope="function")
def source_model():
    yield LrdSampleNetwork()


@pytest.fixture(scope="function")
def source_embedding_cell():
    yield torch.nn.Embedding(64, 32)


@pytest.fixture(scope="function")
def source_decomposer():
    yield low_rank_decompose.Decompose(LrdSampleNetwork())


def test_decompose_embedding_given_valid_when_any_then_pass(source_embedding_cell):
    target_cell = low_rank_decompose.decompose_embedding(source_embedding_cell, 8)
    assert count_parameters(target_cell) == 768


def test_decompose_embedding_given_invalid_when_any_then_error(source_embedding_cell):
    with pytest.raises(ValueError):
        _ = low_rank_decompose.decompose_embedding(source_embedding_cell, "invalid")
    with pytest.raises(ValueError):
        _ = low_rank_decompose.decompose_embedding(source_embedding_cell, 0)


def test_count_parameters_given_valid_when_any_then_pass(source_model):
    assert count_parameters(source_model) == 1038346


def test_get_decomposed_config_given_float_when_any_then_pass(source_model):
    result = {'feature.0': (64, 64), 'inner': 192, 'classifiler.0': 128}
    assert low_rank_decompose.get_decomposed_config(source_model, 0.5) == result

    result = {'embedding.1': 16, 'feature.0': (64, 64), 'feature.2': (32, 32), 'inner': 192, 'classifiler.0': 80}
    assert low_rank_decompose.get_decomposed_config(source_model, 0.5, divisor=16) == result

    result = {'embedding.0': 6, 'embedding.1': 9, 'feature.0': (54, 54), 'feature.2': (18, 18), 'classifiler.1': 6}
    excludes = ['inner', 'classifier.0']
    assert low_rank_decompose.get_decomposed_config(source_model, 0.5, excludes=excludes, divisor=3) == result


def test_get_decomposed_config_given_int_when_any_then_pass(source_model):
    result = {'embedding.1': 16, 'feature.0': (64, 64), 'feature.2': (16, 16), 'inner': 16, 'classifiler.0': 16}
    assert low_rank_decompose.get_decomposed_config(source_model, 13, divisor=16) == result


def test_get_decomposed_config_given_dict_when_any_then_pass(source_model):
    hidden_channels = {'feature.0': (12, 12), 'feature.2': (31, 13), 'inner': 11}
    result = {'feature.0': (16, 16), 'feature.2': (32, 16), 'inner': 192}
    assert low_rank_decompose.get_decomposed_config(source_model, hidden_channels, divisor=16) == result


def test_get_decomposed_config_given_vbmf_when_any_then_pass(source_model):
    result = {'embedding.0': 8, 'embedding.1': 8, 'feature.0': (8, 8), 'feature.2': (8, 8), 'inner': 8, 'classifiler.1': 6}
    excludes = ['classifier.1']
    assert low_rank_decompose.get_decomposed_config(source_model, "vbmf", excludes=excludes, divisor=8) == result


def test_decomposed_network_given_valid_when_any_then_pass(source_model):
    decompose_info = {'embedding.1': 16, 'feature.0': (16, 16), 'feature.2': (16, 16), 'inner': 16, 'classifiler.0': 16}
    new_model = low_rank_decompose.get_decomposed_config(source_model, decompose_info)
    assert count_parameters(new_model) == 60426

    decompose_config = {'feature.0': (32, 16), 'feature.2': (16, 32)}
    new_model = low_rank_decompose.get_decomposed_config(source_model, decompose_config, do_decompose_weight=False)
    assert count_parameters(new_model) == 971786


def test_from_fixed_given_valid_when_any_then_pass(source_decomposer):
    source_decomposer.from_fixed(16)
    assert source_decomposer.decomposer_config == {'feature.0': (64, 64), 'inner': 64, 'classifiler.0': 64}


def test_from_ratio_given_valid_when_any_then_pass(source_decomposer):
    excludes = ['embedding.1', 'classifier.0']
    source_decomposer.from_ratio(0.5, excludes=excludes)
    assert source_decomposer.decomposer_config == {'feature.0': (64, 64), 'inner': 192}


def test_from_dict_given_valid_when_any_then_pass(source_decomposer):
    source_decomposer.from_dict({'feature.0': (32, 16), 'feature.2': (16, 28)}, divisor=16)
    assert source_decomposer.decomposer_config == {'feature.0': (32, 16), 'feature.2': (16, 32)}


def test_from_vbmf_given_valid_when_any_then_pass(source_decomposer):
    excludes = ['embedding.1', 'classifier.0']
    source_decomposer.from_vbmf(excludes=excludes, divisor=8)
    result = {'embedding.0': 8, 'feature.0': (8, 8), 'feature.2': (8, 8), 'inner': 8, 'classifiler.1': 8}
    assert source_decomposer.decomposer_config == result


def test_decompose_network_given_datasets_when_any_then_pass(source_model):
    if hasattr(source_model, "npu"):
        try:
            delattr(source_model, "npu")
        except AttributeError as ee:
            pass
    decompose_info = {'embedding.1': 16, 'feature.0': (16, 16), 'feature.2': (16, 16), 'inner': 16, 'classifiler.0': 16}
    datasets = [[torch.from_numpy(np.random.uniform(size=[2, 16, 16]).astype('int64'))] for _ in range(10)]
    new_model = low_rank_decompose.get_decomposed_config(source_model, decompose_info, datasets=datasets)
    assert count_parameters(new_model) == 60426
