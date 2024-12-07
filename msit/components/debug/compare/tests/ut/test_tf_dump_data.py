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
# Wcmp_process.pyITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pytest

from msquickcmp.adapter_cli.args_adapter import CmpArgsAdapter

try:
    from msquickcmp.tf.tf_save_model_dump_data import TfSaveModelDumpData
except:
    TfSaveModelDumpData = None


@pytest.mark.skipif(TfSaveModelDumpData is None, reason="import tensorflow error")
def test_split_input_shape():
    input_shapes = "input_1:16,32,32,3;input_2:1,16,16,32"
    input_shape_list = TfSaveModelDumpData.split_input_shape(input_shapes)
    expect_input_shape_list = [("input_1", [16, 32, 32, 3]), ("input_2", [1, 16, 16, 32])]
    assert expect_input_shape_list == input_shape_list
