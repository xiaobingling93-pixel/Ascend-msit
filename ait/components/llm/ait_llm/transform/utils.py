import os
from ait_llm.common.log import logger
from collections import namedtuple

SCENARIOS = namedtuple("SCENARIOS", ["torch_to_float_atb", "float_atb_to_quant_atb"])("torch_to_float_atb", "float_atb_to_quant_atb")


def get_transform_scenario(source_path):
    if os.path.isfile(source_path) and source_path.endswith(".cpp"):  # Single cpp input
        return SCENARIOS.float_atb_to_quant_atb

    cur_items = os.listdir(source_path)
    if "config.json" in cur_items and any([ii.endswith(".py") for ii in cur_items]):
        return SCENARIOS.torch_to_float_atb
    elif any([ii.endswith(".cpp") for ii in cur_items]):
        return SCENARIOS.float_atb_to_quant_atb
    else:
        return None
