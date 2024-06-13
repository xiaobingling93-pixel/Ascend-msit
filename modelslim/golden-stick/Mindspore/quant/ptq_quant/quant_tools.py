import os
import mindspore as ms
from mindspore_gs.ptq import RoundToNearest as RTN
from .quant_config import QuantConfig
from .llm_ptq_utils import gen_fake_inputs

class Calibrator(object):
    def __init__(self, model,
                 cfg: QuantConfig):
        self.network = model
        self.cfg = cfg
        self.ptq = RTN(config=self.cfg)
    
    def run(self):
        qnet = self.ptq.apply(self.network.model)
        qnet = self.ptq.convert(qnet)
        self.network.model = qnet
    
    def save(self, output_path):
        if not isinstance(output_path, str):
            raise ValueError("output_path must be a string")
        os.makedirs(output_path, exist_ok=True)

        ms.save_checkpoint(self.network.parameters_dict(), os.path.join(output_path, "w8a16.ckpt"),
                       choice_func=lambda x: "key_cache" not in x and "value_cache" not in x)