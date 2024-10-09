import os
import json
import stat
import pandas as pd
import pytest
import torch
import numpy as np
from components.debug.compare.msquickcmp.mie_torch.mietorch_comp import MIETorchCompare
from components.debug.compare.utils.ge_dump_reader import GEDumpFileReader
from components.debug.compare.utils.torch_dump_reader import TorchDumpFileReader


FAKE_GOLDEN_DATA_PATH = "test_resource"
FAKE_MY_DATA_PATH = "test_resource"
FAKE_OPS_JSON = "test_resource"
FAKE_RT_OP_MAPPING_JSON_PATH = "test_resource/mindie_rt_op_mapping.json"
FAKE_TORCH_OP_MAPPING_JSON_PATH = "test_resource/mindie_torch_op_mapping.json"


flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL  
mode = stat.S_IWUSR | stat.S_IRUSR  


@pytest.fixture(scope='module')
def rt_op_mapping_file():
    rt_op_mapping_data = [{
        "fusion_op": "Conv2D_5Relu_8",
        "fusion_ops": [
            "Conv2D_5",
            "Relu_8"
        ],
        "ge_op": "Conv2D_5",
        "id": 4,
        "rt_layer": "CONVOLUTION_5"
    },
    {
        "fusion_op": "Conv2D_5Relu_8",
        "fusion_ops": [
            "Conv2D_5",
            "Relu_8"
        ],
        "ge_op": "Conv2D_5",
        "id": 5,
        "rt_layer": "ACTIVATION_6"
    }]
    current_file_path = os.path.dirname(__file__)
    global FAKE_RT_OP_MAPPING_JSON_PATH
    FAKE_RT_OP_MAPPING_JSON_PATH = os.path.join(current_file_path, FAKE_RT_OP_MAPPING_JSON_PATH)
    with os.fdopen(os.open(FAKE_RT_OP_MAPPING_JSON_PATH, flags, mode), 'w') as f:
        json.dump(rt_op_mapping_data, f, indent=4)

    yield FAKE_RT_OP_MAPPING_JSON_PATH
    
    if os.path.exists(FAKE_RT_OP_MAPPING_JSON_PATH):
        os.remove(FAKE_RT_OP_MAPPING_JSON_PATH)


@pytest.fixture(scope='module')
def torch_op_mapping_file():
    torch_op_mapping_data = [{
        "jit_node": "%input.9 : Float(1, 64, 56, 56) = aten::_convolution(%input.7, \
        %self.layer1.0.conv1.weight.1_fused_bn, %9_fused_bn.3, %6, %5, %6, %11, %5, \
        %10, %11, %11, %12, %12), scope: __module.layer1/__module.layer1.0/__module.layer1.0.conv1 \
        # /usr/local/python3.10.2/lib/python3.10/site-packages/torch/nn/modules/conv.py:456:0",
        "rt_layer": "CONVOLUTION_5"
    },
    {
        "jit_node": "%968 : Float(1, 64, 56, 56) = aten::relu(%input.9), scope: \
        __module.layer1/__module.layer1.0/__module.layer1.0.relu \
        # /usr/local/python3.10.2/lib/python3.10/site-packages/torch/nn/functional.py:1469:0",
        "rt_layer": "ACTIVATION_6"
    }]
    current_file_path = os.path.dirname(__file__)
    global FAKE_TORCH_OP_MAPPING_JSON_PATH
    FAKE_TORCH_OP_MAPPING_JSON_PATH = os.path.join(current_file_path, FAKE_TORCH_OP_MAPPING_JSON_PATH)
    with os.fdopen(os.open(FAKE_TORCH_OP_MAPPING_JSON_PATH, flags, mode), 'w') as f:
        json.dump(torch_op_mapping_data, f, indent=4)

    yield FAKE_TORCH_OP_MAPPING_JSON_PATH

    if os.path.exists(FAKE_TORCH_OP_MAPPING_JSON_PATH):
        os.remove(FAKE_TORCH_OP_MAPPING_JSON_PATH)
    

def test_mindie_torch_compare(mocker, rt_op_mapping_file, torch_op_mapping_file):
    current_file_path = os.path.dirname(__file__)
    global FAKE_OPS_JSON
    FAKE_OPS_JSON = os.path.join(current_file_path, FAKE_OPS_JSON)
    out_path = os.path.join(current_file_path, "test_resource")
    copmarer = MIETorchCompare(FAKE_GOLDEN_DATA_PATH, FAKE_MY_DATA_PATH, FAKE_OPS_JSON, out_path)

    test_data = torch.ones((1, 64, 56, 56), dtype=torch.float32)
    test_data[0, 0, 0, 0] = 10
    mocker_function_cpu = mocker.patch.object(TorchDumpFileReader, 'get_tensor')
    mocker_function_cpu.return_value = test_data
    mocker_function_npu = mocker.patch.object(GEDumpFileReader, 'get_tensor')
    mocker_function_npu.return_value = test_data.numpy()

    res = copmarer.compare()
    csv_file_path = os.path.join(out_path, "comparison_results.csv")
    assert os.path.isfile(csv_file_path)
    df = pd.read_csv(csv_file_path)
    assert df.loc[0, "cosine_similarity"] == 1.0
    new_json_file = os.path.join(current_file_path, "test_resource", "op_map_updated.json")
    
    if os.path.exists(csv_file_path):
        os.remove(csv_file_path)
    if os.path.exists(new_json_file):
        os.remove(new_json_file)