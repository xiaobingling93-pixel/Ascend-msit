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

import os
import shutil
from test.resources.sample_net_torch import ThreeLinearTorchModel_for_Sparse
import torch

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator as SparseQuantCalibrator
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import QuantConfig as SparseQuantConfig


TEST_SAVE_PATH = "sparse_quant_save_path"

class Config:
    def __init__(self, torch_dtype):
        self.torch_dtype = torch_dtype


def test_sparse_quant():
    os.makedirs(TEST_SAVE_PATH, exist_ok=True)
    os.chdir(TEST_SAVE_PATH)
    model = ThreeLinearTorchModel_for_Sparse
    model.config = Config(torch_dtype=torch.float16)
    dataset_calib = [[torch.randn(1, 256, 256)]]
    w_bit = 4
    a_bit = 8
    act_method = 3
    pr = 2.0
    fraction = 0.011
    nonuniform = False
    mm_tensor = False

    quant_config = SparseQuantConfig(
        w_bit=w_bit,
        a_bit=a_bit,
        dev_type='cpu',
        act_method=act_method,
        pr=pr,
        fraction=fraction,
        nonuniform=nonuniform,
        mm_tensor=mm_tensor,
        co_sparse=True
    )
    calibrator = SparseQuantCalibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')
    calibrator.run(int_infer=False)
    calibrator.save(TEST_SAVE_PATH, save_type=["numpy", "safe_tensor"])
    if os.path.exists(TEST_SAVE_PATH):
        shutil.rmtree(TEST_SAVE_PATH)


def test_sparse_quant_change_int_infer():
    os.makedirs(TEST_SAVE_PATH, exist_ok=True)
    os.chdir(TEST_SAVE_PATH)
    model = ThreeLinearTorchModel_for_Sparse
    model.config = Config(torch_dtype=torch.float16)
    dataset_calib = [[torch.randn(1, 256, 256)]]
    w_bit = 4
    a_bit = 8
    act_method = 3
    pr = 2.0
    fraction = 0.011
    nonuniform = False
    mm_tensor = False

    quant_config = SparseQuantConfig(
        w_bit=w_bit,
        a_bit=a_bit,
        dev_type='cpu',
        act_method=act_method,
        pr=pr,
        fraction=fraction,
        nonuniform=nonuniform,
        mm_tensor=mm_tensor,
        co_sparse=True
    )
    calibrator = SparseQuantCalibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')
    calibrator.run(int_infer=True)
    calibrator.save(TEST_SAVE_PATH, save_type=["numpy", "safe_tensor"])
    if os.path.exists(TEST_SAVE_PATH):
        shutil.rmtree(TEST_SAVE_PATH)


def test_sparse_quant_change_act_method_1():
    os.makedirs(TEST_SAVE_PATH, exist_ok=True)
    os.chdir(TEST_SAVE_PATH)
    model = ThreeLinearTorchModel_for_Sparse
    model.config = Config(torch_dtype=torch.float16)
    dataset_calib = [[torch.randn(1, 256, 256)]]
    w_bit = 4
    a_bit = 8
    act_method = 1
    pr = 2.0
    fraction = 0.011
    nonuniform = False
    mm_tensor = False

    quant_config = SparseQuantConfig(
        w_bit=w_bit,
        a_bit=a_bit,
        dev_type='cpu',
        act_method=act_method,
        pr=pr,
        fraction=fraction,
        nonuniform=nonuniform,
        mm_tensor=mm_tensor,
        co_sparse=True
    )
    calibrator = SparseQuantCalibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')
    calibrator.run(int_infer=False)
    calibrator.save(TEST_SAVE_PATH, save_type=["numpy", "safe_tensor"])
    if os.path.exists(TEST_SAVE_PATH):
        shutil.rmtree(TEST_SAVE_PATH)


def test_sparse_quant_change_act_method_2():
    os.makedirs(TEST_SAVE_PATH, exist_ok=True)
    os.chdir(TEST_SAVE_PATH)
    model = ThreeLinearTorchModel_for_Sparse
    model.config = Config(torch_dtype=torch.float16)
    dataset_calib = [[torch.randn(1, 256, 256)]]
    w_bit = 4
    a_bit = 8
    act_method = 2
    pr = 2.0
    fraction = 0.011
    nonuniform = False
    mm_tensor = False

    quant_config = SparseQuantConfig(
        w_bit=w_bit,
        a_bit=a_bit,
        dev_type='cpu',
        act_method=act_method,
        pr=pr,
        fraction=fraction,
        nonuniform=nonuniform,
        mm_tensor=mm_tensor,
        co_sparse=True
    )
    calibrator = SparseQuantCalibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')
    calibrator.run(int_infer=False)
    calibrator.save(TEST_SAVE_PATH, save_type=["numpy", "safe_tensor"])
    if os.path.exists(TEST_SAVE_PATH):
        shutil.rmtree(TEST_SAVE_PATH)


def test_sparse_quant_change_nonuniform():
    os.makedirs(TEST_SAVE_PATH, exist_ok=True)
    os.chdir(TEST_SAVE_PATH)
    model = ThreeLinearTorchModel_for_Sparse
    model.config = Config(torch_dtype=torch.float16)
    dataset_calib = [[torch.randn(1, 256, 256)]]
    w_bit = 4
    a_bit = 8
    act_method = 3
    pr = 2.0
    fraction = 0.011
    nonuniform = True
    mm_tensor = False

    quant_config = SparseQuantConfig(
        w_bit=w_bit,
        a_bit=a_bit,
        dev_type='cpu',
        act_method=act_method,
        pr=pr,
        fraction=fraction,
        nonuniform=nonuniform,
        mm_tensor=mm_tensor,
        co_sparse=True
    )
    calibrator = SparseQuantCalibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')
    calibrator.run(int_infer=True)
    calibrator.save(TEST_SAVE_PATH, save_type=["numpy", "safe_tensor"])
    if os.path.exists(TEST_SAVE_PATH):
        shutil.rmtree(TEST_SAVE_PATH)


def test_sparse_quant_change_mm_tensor():
    os.makedirs(TEST_SAVE_PATH, exist_ok=True)
    os.chdir(TEST_SAVE_PATH)
    model = ThreeLinearTorchModel_for_Sparse
    model.config = Config(torch_dtype=torch.float16)
    dataset_calib = [[torch.randn(1, 256, 256)]]
    w_bit = 4
    a_bit = 8
    act_method = 3
    pr = 2.0
    fraction = 0.011
    nonuniform = False
    mm_tensor = True

    quant_config = SparseQuantConfig(
        w_bit=w_bit,
        a_bit=a_bit,
        dev_type='cpu',
        act_method=act_method,
        pr=pr,
        fraction=fraction,
        nonuniform=nonuniform,
        mm_tensor=mm_tensor,
        co_sparse=True
    )
    calibrator = SparseQuantCalibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')
    calibrator.run(int_infer=False)
    calibrator.save(TEST_SAVE_PATH, save_type=["numpy", "safe_tensor"])
    if os.path.exists(TEST_SAVE_PATH):
        shutil.rmtree(TEST_SAVE_PATH)


def test_sparse_quant_lowbit():
    os.makedirs(TEST_SAVE_PATH, exist_ok=True)
    os.chdir(TEST_SAVE_PATH)
    model = ThreeLinearTorchModel_for_Sparse
    model.config = Config(torch_dtype=torch.float16)
    dataset_calib = [[torch.randn(1, 256, 256)]]
    w_bit = 4
    a_bit = 8
    act_method = 3
    do_smooth = False
    use_sigma = False
    sigma_factor = 3.0

    quant_config = SparseQuantConfig(
        w_bit=w_bit,
        a_bit=a_bit,
        dev_type='cpu',
        act_method=act_method,
        do_smooth=do_smooth,
        use_sigma=use_sigma,
        sigma_factor=sigma_factor,
        is_lowbit=True
    )
    calibrator = SparseQuantCalibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')
    calibrator.run(int_infer=False)
    calibrator.save(TEST_SAVE_PATH, save_type=["numpy", "safe_tensor"])
    if os.path.exists(TEST_SAVE_PATH):
        shutil.rmtree(TEST_SAVE_PATH)


def test_sparse_quant_lowbit_change_do_smooth():
    os.makedirs(TEST_SAVE_PATH, exist_ok=True)
    os.chdir(TEST_SAVE_PATH)
    model = ThreeLinearTorchModel_for_Sparse
    model.config = Config(torch_dtype=torch.float16)
    dataset_calib = [[torch.randn(1, 256, 256)]]
    w_bit = 4
    a_bit = 8
    act_method = 3
    do_smooth = True
    use_sigma = False
    sigma_factor = 3.0

    quant_config = SparseQuantConfig(
        w_bit=w_bit,
        a_bit=a_bit,
        dev_type='cpu',
        act_method=act_method,
        do_smooth=do_smooth,
        use_sigma=use_sigma,
        sigma_factor=sigma_factor,
        is_lowbit=True
    )
    calibrator = SparseQuantCalibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')
    calibrator.run(int_infer=False)
    calibrator.save(TEST_SAVE_PATH, save_type=["numpy", "safe_tensor"])
    if os.path.exists(TEST_SAVE_PATH):
        shutil.rmtree(TEST_SAVE_PATH)


def test_sparse_quant_lowbit_change_use_sigma():
    os.makedirs(TEST_SAVE_PATH, exist_ok=True)
    os.chdir(TEST_SAVE_PATH)
    model = ThreeLinearTorchModel_for_Sparse
    model.config = Config(torch_dtype=torch.float16)
    dataset_calib = [[torch.randn(1, 256, 256)]]
    w_bit = 4
    a_bit = 8
    act_method = 3
    do_smooth = True
    use_sigma = True
    sigma_factor = 3.0

    quant_config = SparseQuantConfig(
        w_bit=w_bit,
        a_bit=a_bit,
        dev_type='cpu',
        act_method=act_method,
        do_smooth=do_smooth,
        use_sigma=use_sigma,
        sigma_factor=sigma_factor,
        is_lowbit=True
    )
    calibrator = SparseQuantCalibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')
    calibrator.run(int_infer=False)
    calibrator.save(TEST_SAVE_PATH, save_type=["numpy", "safe_tensor"])
    if os.path.exists(TEST_SAVE_PATH):
        shutil.rmtree(TEST_SAVE_PATH)