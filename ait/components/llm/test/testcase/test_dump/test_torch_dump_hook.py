import os
import shutil
from glob import glob

import torch
from torch import nn

from ait_llm import DumpConfig
from ait_llm import register_hook


MODEL_NAME_LIST = ["root", "root.ln"]
DUMP_PATH = "./ait_dump"


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
    except_output_path = glob(os.pth.join(DUMP_PATH, "*", "torch_tensors"))[0]
    for name in MODEL_NAME_LIST:
        except_input_path = os.path.join(except_output_path, "cpu_" + str(os.getpid()), "0", name, "input_0.pth")
        except_output_path = os.path.join(except_output_path, "cpu_" + str(os.getpid()), "0", name, "output.pth")
        assert os.path.exists(except_input_path)
        assert os.path.exists(except_output_path)
    topo_path = os.path.join(except_output_path, "cpu_" + str(os.getpid()), "model_tree.json")
    assert os.path.exists(topo_path)
        
    if os.path.exists(DUMP_PATH):
        shutil.rmtree(DUMP_PATH)
