import os
import shutil
from glob import glob

import torch
from torch import nn

from msit_llm import DumpConfig
from msit_llm import register_hook

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


def test_hook_when_tp_default_then_save_inputs():
    model = SampleModel()
    dump_config = DumpConfig(dump_path=DUMP_PATH)
    register_hook(model, dump_config)
    x = torch.randn(4, 4)
    model(x)
    output_path_prefix = glob(os.path.join(DUMP_PATH, "*", "torch_tensors"))[0]
    for name in MODEL_NAME_LIST:
        except_input_path = os.path.join(output_path_prefix, "cpu_" + str(os.getpid()), "0", name, "input_0.pth")
        except_output_path = os.path.join(output_path_prefix, "cpu_" + str(os.getpid()), "0", name, "output.pth")
        assert os.path.exists(except_input_path)
        assert os.path.exists(except_output_path)
    topo_path = os.path.join(output_path_prefix, "cpu_" + str(os.getpid()), "model_tree.json")
    assert os.path.exists(topo_path)
        
    if os.path.exists(DUMP_PATH):
        shutil.rmtree(DUMP_PATH)
