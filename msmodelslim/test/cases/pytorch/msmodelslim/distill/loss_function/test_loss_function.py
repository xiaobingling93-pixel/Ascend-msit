# Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
try:
    from torch.tensor import Tensor
except ModuleNotFoundError as ee:
    from torch import Tensor
import pytest

from ascend_utils.pytorch.knowledge_distill.distill_losses_func_torch import DISTILL_LOSS_FUNC_TORCH

@pytest.fixture()
def tensor_s():
    yield Tensor([[1.0], [1.0]])

@pytest.fixture()
def tensor_t():
    yield Tensor([[1.0], [1.0]])

def test_kd_cross_entropy_given_valid_when_any_then_pass(tensor_s, tensor_t):
    loss_func_kd_cross_entropy = DISTILL_LOSS_FUNC_TORCH["KDCrossEntropy"]()
    loss_func_kd_cross_entropy(tensor_s, tensor_t)