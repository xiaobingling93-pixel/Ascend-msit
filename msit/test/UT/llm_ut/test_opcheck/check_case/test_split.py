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
from unittest.mock import MagicMock
import pytest


# Mocking the OperationTest class to avoid errors
class MockOperationTest:
    def __init__(self, *args, **kwargs):
        self.op_param = kwargs.get('op_param', {})

    @staticmethod
    def validate_param(self, *args, **kwargs):
        # Mock validation; return True for simplicity
        return True

    def execute(self):
        # Mock execution; do nothing
        pass


@pytest.fixture(scope="function")
def import_add_module():
    backup = {}
    for mod in ['torch_npu']:
        if mod in sys.modules:
            backup[mod] = sys.modules[mod]
    mock_torch_npu = MagicMock()
    sys.modules['torch_npu'] = mock_torch_npu
    from msit_llm.opcheck.check_case.split import OpcheckAddOperation
    OpcheckAddOperation.__bases__ = (MockOperationTest,)
    functions = {
        "OpcheckAddOperation": OpcheckAddOperation
    }
    yield functions
    
    for mod, module_obj in backup.items():
        sys.modules[mod] = module_obj
    for mod in ['torch_npu']:
        if mod not in backup and mod in sys.modules:
            del sys.modules[mod]


def test_test_method_given_valid_params_when_called_then_execute_called(import_add_module):
    OpcheckAddOperation = import_add_module['OpcheckAddOperation']
    op = OpcheckAddOperation(op_param={'splitDim': 0, 'splitNum': 2})
    op.execute = MagicMock()

    op.test()
    
    op.execute.assert_called_once()
