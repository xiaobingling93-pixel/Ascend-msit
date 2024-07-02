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
import shutil
import subprocess
import stat

import pytest
import torch
import acl


FAKE_OM_MODEL_PATH = "fake_test_onnx_model.om"
OM_OUT_PATH = FAKE_OM_MODEL_PATH.replace(".om", "")


FAKE_ONNX_MODEL_PATH = "fake_msquickcmp_test_onnx_model.onnx"
OUT_PATH = FAKE_ONNX_MODEL_PATH.replace(".onnx", "")
INPUT_SHAPE = (1, 3, 32, 32)

WRITE_FLAGS = os.O_WRONLY | os.O_CREAT  # 注意根据具体业务的需要设置文件读写方式
WRITE_MODES = stat.S_IWUSR | stat.S_IRUSR  # 注意根据具体业务的需要设置文件权限


def get_cann_path():
    result = subprocess.run(['which', 'atc'], stdout=subprocess.PIPE)
    atc_path = result.stdout.decode('utf-8').strip()
    cann_path = atc_path[:-8]
    return cann_path


CANN_PATH = get_cann_path()


@pytest.fixture(scope="module", autouse=True)
def width_onnx_model():
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 32, 1, 1),
        torch.nn.BatchNorm2d(32),
        torch.nn.Conv2d(32, 32, 3, 2),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(32, 32),
        torch.nn.Linear(32, 10),
    )
    torch.onnx.export(model, torch.ones(INPUT_SHAPE), FAKE_ONNX_MODEL_PATH)
    os.chmod(FAKE_ONNX_MODEL_PATH, 0o750)
    yield FAKE_ONNX_MODEL_PATH

    if os.path.exists(FAKE_ONNX_MODEL_PATH):
        os.remove(FAKE_ONNX_MODEL_PATH)
    if os.path.exists(OUT_PATH):
        shutil.rmtree(OUT_PATH)


@pytest.fixture(scope="module", autouse=True)
def fake_om_model(width_onnx_model):
    if not os.path.exists(FAKE_OM_MODEL_PATH):
        cmd = 'atc --model={}  --framework=5 --output={} \
            --soc_version={}'.format(width_onnx_model,
                                     OM_OUT_PATH,
                                     acl.get_soc_name())
        subprocess.run(cmd.split(), shell=False)

    yield FAKE_OM_MODEL_PATH

    if os.path.exists(FAKE_OM_MODEL_PATH):
        os.remove(FAKE_OM_MODEL_PATH)


def test_basic_usage_then_pass(width_onnx_model, fake_om_model):
    cmd = 'msit debug compare -gm {} -om {} -c {} -o {} --max-cmp-size 1024'.format(width_onnx_model,
                                                    fake_om_model,
                                                    CANN_PATH,
                                                    OUT_PATH)
    if not os.path.exists(OUT_PATH):
        os.mkdir(OUT_PATH, 0o750)

    ret = os.system(cmd)
    assert ret == 0

    assert len(os.listdir(OUT_PATH)) > 0

    if os.path.exists(OUT_PATH):
        shutil.rmtree(OUT_PATH)