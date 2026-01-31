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
from unittest.mock import patch, MagicMock
import pytest
import torch
from mock_operation_test import MockOperationTest

@pytest.fixture(scope="function")
def import_rope_grad_module():
    backup = {}
    for mod in ['torch_npu']:
        if mod in sys.modules:
            backup[mod] = sys.modules[mod]
    mock_torch_npu = MagicMock()
    sys.modules['torch_npu'] = mock_torch_npu
    
    from msit_llm.opcheck.check_case.rope_grad import OpcheckRopeGradOperation
    OpcheckRopeGradOperation.__bases__ = (MockOperationTest,)
    
    yield {"OpcheckRopeGradOperation": OpcheckRopeGradOperation}
    
    for mod, module_obj in backup.items():
        sys.modules[mod] = module_obj
    for mod in ['torch_npu']:
        if mod not in backup and mod in sys.modules:
            del sys.modules[mod]

def test_golden_calc_given_empty_in_tensors_when_execution_then_raise_exception(import_rope_grad_module):
    OpcheckRopeGradOperation = import_rope_grad_module['OpcheckRopeGradOperation']
    op = OpcheckRopeGradOperation()
    op.op_param = {'qSeqLen': [128, 128]}
    with pytest.raises(IndexError):
        op.golden_calc([])

def test_golden_calc_given_small_qSeqLen_when_execution_then_correct_result(import_rope_grad_module):
    OpcheckRopeGradOperation = import_rope_grad_module['OpcheckRopeGradOperation']
    test_cases = [
        {'qSeqLen': [64, 64]},
        {'qSeqLen': [32, 32, 32, 32]}
    ]
    
    for case in test_cases:
        op = OpcheckRopeGradOperation()
        op.op_param = case
        in_tensors = [
            torch.randn(1, 128, 128),
            torch.randn(1, 128, 128),
            torch.randn(128, 128),
            torch.randn(128, 128)
        ]
        result = op.golden_calc(in_tensors)
        assert result[0].shape == (1, 128, 128)
        assert result[1].shape == (1, 128, 128)

def test_golden_calc_given_small_values_when_execution_then_correct_result(import_rope_grad_module):
    OpcheckRopeGradOperation = import_rope_grad_module['OpcheckRopeGradOperation']
    test_cases = [
        {'qSeqLen': [32, 32, 32, 32], 'scale': 0.001, 'tensor_idx': [2, 3]},  # cos/sin
        {'qSeqLen': [32, 32, 32, 32], 'scale': 0.001, 'tensor_idx': [0, 1]},  # q_grad/k_grad
        {'qSeqLen': [32, 32, 32, 32], 'scale': 0.001, 'tensor_idx': [0, 1, 2, 3]}  # all
    ]
    
    for case in test_cases:
        op = OpcheckRopeGradOperation()
        op.op_param = {'qSeqLen': case['qSeqLen']}
        in_tensors = [
            torch.randn(1, 128, 128),
            torch.randn(1, 128, 128),
            torch.randn(128, 128),
            torch.randn(128, 128)
        ]
        for idx in case['tensor_idx']:
            in_tensors[idx] *= case['scale']
            
        result = op.golden_calc(in_tensors)
        assert result[0].shape == (1, 128, 128)
        assert result[1].shape == (1, 128, 128)