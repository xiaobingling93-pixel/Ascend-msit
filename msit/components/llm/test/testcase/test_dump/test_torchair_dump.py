from glob import glob
import os
import numpy as np
import torch
import torch.nn as nn
import torch_npu
import torchair as tng
import torchvision
from msit_llm.dump import torchair_dump
from components.llm.msit_llm.common.constant import GLOBAL_AIT_DUMP_PATH

MODEL_NAME_LIST = ["root", "root.ln"]
DUMP_PATH = f"./{GLOBAL_AIT_DUMP_PATH}"


class SampleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(4)

    def forward(self, x):
        y = self.ln(x)
        z = y + y
        return z


def test_dump_torchair_token_layer():
    target_dtype = torch.float16
    model = SampleModel().eval().to(target_dtype).npu()

    aa = torch.ones(1, 3, 224, 224).to(target_dtype).npu()
    dump_token = [0, 2, 5]
    dump_layer = ["Add_8", "BNInfer_14"]
    config = torchair_dump.get_ge_dump_config(dump_token=dump_token, dump_layer=dump_layer, dump_path=DUMP_PATH)
    npu_backend = tng.get_npu_backend(compiler_config=config)
    model = torch.compile(model, backend=npu_backend, dynamic=True)
    with torch.no_grad():
        for _ in range(10):
            shape = model(aa).shape

    output_path_prefix = glob(os.path.join(DUMP_PATH, "**", "graph_*", "*"), recursive=True)
    assert len(os.listdir(output_path_prefix[0])) == len(dump_token)

    for token in dump_token:
        token_path = os.path.join(output_path_prefix[0], str(token))
        assert os.path.exists(token_path)
        for layer in dump_layer:
            layer_path = glob(os.path.join(token_path, "*." + layer + ".*"))
            assert len(layer_path) > 0
        
    if os.path.exists(DUMP_PATH):
        shutil.rmtree(DUMP_PATH)
