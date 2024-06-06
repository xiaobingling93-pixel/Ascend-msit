import os
import time
from ait_llm.transform.torch_to_float_atb import utils
from ait_llm.transform.model_parser import parser

def float_layer_cpp_gen(model, save_name=None, save_dir=None):
    # NotImplemented
    rr = ""

    save_name = utils.init_save_name(save_name) + ".cpp"
    save_dir = utils.init_save_dir(model_name_lower if save_dir is None else save_dir, sub_dir="layer")
    save_path = os.path.join(save_dir, save_file)
    with open(save_path, "w") as ff:
        ff.write(rr)
    return save_path, rr