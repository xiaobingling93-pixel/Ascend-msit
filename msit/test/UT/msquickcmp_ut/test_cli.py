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
import pytest
from unittest.mock import patch, MagicMock
from msquickcmp.adapter_cli.args_adapter import CmpArgsAdapter
from msquickcmp.common import utils


@pytest.fixture(scope="module", autouse=True)
def compare_cli() -> None:
    cmp_args = CmpArgsAdapter(
        gold_model="./fake.onnx",
        om_model="./fake.om",
        input_data_path="",
        out_path="",
        input_shape="",
        device='0',
        output_size="",
        output_nodes="",
        advisor=False,
        dym_shape_range="",
        dump=True,
        bin2npy=False,
    )
    yield cmp_args


def test_args_invalid_path_err(compare_cli):
    mock_acl = MagicMock()
    with pytest.raises(utils.AccuracyCompareException) as error, \
         patch.dict('sys.modules', {
             'acl': mock_acl,
         }):
        from msquickcmp.cmp_process import cmp_process
        cmp_process(compare_cli, True)

    assert error.value.error_info == utils.ACCURACY_COMPARISON_INVALID_PATH_ERROR
