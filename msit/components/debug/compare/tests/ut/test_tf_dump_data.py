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

import pytest

from msquickcmp.tf.tf_save_model_dump_data import TfSaveModelDumpData


@pytest.fixture(scope='module', autouse=True)
def test_parse_ops_name_from_om_json():
    om_json = "./test_resource/om/model.json"
    expect_ops_name = ['boxes_all', 'scores_all', 'Slice_1218', 'Slice_1218', 'Gather_1221']
    ops_name = TfSaveModelDumpData.parse_ops_name_from_om_json(om_json)
    assert expect_ops_name == ops_name


@pytest.fixture(scope='module', autouse=True)
def test_split_input_shape():
    input_shapes = "input_1:16,32,32,3;input_2:1,16,16,32"
    input_shape_list = TfSaveModelDumpData.split_input_shape(input_shapes)
    expect_input_shape_list = [('input_1', [16, 32, 32, 3]), ('input_2', [1, 16, 16, 32])]
    assert expect_input_shape_list == input_shape_list



