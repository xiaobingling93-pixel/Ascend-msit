# Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import os

import onnx
import torch
import pytest
import numpy as np
import torch.nn as nn

from ascend_utils.common.security.path import set_file_stat
from modelslim.pytorch.quant.ptq_tools import Calibrator, QuantConfig

ONNX_MODEL = "fake_test_ptq_pytorch_tools"
ONNX_QUANT_MODEL_PATH = "fake_test_ptq_pytorch_tools_quant.onnx"
CONV2_NAME = "conv2"


class DummyConv(nn.Module):
    def __init__(self):
        super(DummyConv, self).__init__()
        self.conv1 = nn.Conv2d(3, 4, 3)
        self.conv2 = nn.Conv2d(4, 4, 3)
        self.conv3 = nn.Conv2d(4, 4, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


@pytest.fixture(scope="function")
def generate_model():
    model = DummyConv()
    yield model


def get_onnx_params():
    onnx_model = onnx.load(ONNX_QUANT_MODEL_PATH)
    graph = onnx_model.graph

    offset_list = []
    scale_list = []

    for node in graph.node:
        if node.op_type == 'AscendQuant':
            for attr in node.attribute:
                if attr.name == 'offset':
                    current_offset = attr.f
                    offset_list.append(current_offset)
                elif attr.name == 'scale':
                    current_scale = attr.f
                    scale_list.append(current_scale)
    quant_weight_dict = {}
    conv_weight_names = ['conv1.weight_0', 'conv2.weight_1']
    for initializer in onnx_model.graph.initializer:
        for name in conv_weight_names:
            if initializer.name == name:
                weight_tensor = initializer
                weight_value = np.frombuffer(weight_tensor.raw_data, dtype=np.int8)
                weight_value = weight_value.reshape(weight_tensor.dims)
                quant_weight_dict[name] = weight_value
    return offset_list, scale_list, quant_weight_dict


def test_pytorch_ptq_given_model_then_pass(generate_model):
    config = QuantConfig(
        disable_names=[],
        input_shape=[1, 3, 32, 32],
        amp_num=0
    )
    calib = Calibrator(generate_model, config)
    calib.run()
    input_scale, input_offset, _, _, quant_weight = calib.get_quant_params()
    calib.export_quant_onnx(ONNX_MODEL, "./", ["input.1"])
    set_file_stat(ONNX_QUANT_MODEL_PATH, stat_mode="440")
    offset_list, scale_list, quant_weight_dict = get_onnx_params()

    assert input_offset.get(CONV2_NAME).item() == offset_list[1]
    assert round(input_scale.get(CONV2_NAME).item(), 5) == round(1 / scale_list[1], 5)
    assert torch.equal(quant_weight.get('conv1').int(),
                       torch.from_numpy(quant_weight_dict.get('conv1.weight_0')).int())
    assert torch.equal(quant_weight.get(CONV2_NAME).int(),
                       torch.from_numpy(quant_weight_dict.get('conv2.weight_1')).int())

    if os.path.exists(ONNX_MODEL):
        os.remove(ONNX_MODEL)

    if os.path.exists(ONNX_QUANT_MODEL_PATH):
        os.remove(ONNX_QUANT_MODEL_PATH)
