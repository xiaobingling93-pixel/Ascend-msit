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
import pytest
import subprocess
import shutil

import torch
from torch import nn

try:
    import acl
except:
    acl = None


TEST_ONNX_FILE = "test.onnx"
TEST_OM_FILE = "test.om"
OUTPUT_PATH = "output_datas/"


class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, (3, 3))

    def forward(self, x):
        return self.conv(x)


@pytest.fixture(scope="module", autouse=True)
def basic_onnx_model():
    model = TestModel()
    dummy_input = torch.randn((1, 3, 32, 32))
    torch.onnx.export(model, dummy_input, TEST_ONNX_FILE)
    os.chmod(TEST_ONNX_FILE, 0o640)

    soc_version = acl.get_soc_name()
    om_name = os.path.splitext(TEST_OM_FILE)[0]  # Get rid of .om
    subprocess.run(
        ["atc", "--model", TEST_ONNX_FILE, "--soc-version", soc_version, "--output", om_name, "--framework", "5"]
    )

    yield

    if os.path.exists(TEST_ONNX_FILE):
        os.remove(TEST_ONNX_FILE)
    if os.path.exists(TEST_OM_FILE):
        os.remove(TEST_OM_FILE)
    if os.path.exists(OUTPUT_PATH):
        shutil.rmtree(OUTPUT_PATH)


def test_default_cmd():
    app_cmd = "'msit benchmark -om {}'".format(TEST_OM_FILE)
    cmd = "msit profile --application {} -o {}".format(app_cmd, OUTPUT_PATH)
    ret = os.system(cmd)
    assert ret == 0


def test_not_default_cmd():
    app_cmd = "'msit benchmark -om {}'".format(TEST_OM_FILE)
    cmd = "msit profile --application {} -o {} --model-execution {} --sys-hardware-mem {} \
	--sys-profiling {} --sys-pid-profiling {} --dvpp-profiling {} --runtime-api {} \
	--task-time {} --aicpu {}".format(
        app_cmd, OUTPUT_PATH, "on", "on", "off", "off", "on", "on", "on", "off", "off"
    )
    ret = os.system(cmd)
    assert ret == 0
