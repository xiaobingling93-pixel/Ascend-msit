import os
import time
from msit_llm.transform.torch_to_float_atb import utils

def router_py_gen(parsed_model, save_name=None, save_dir=None):
    from msit_llm.transform.torch_to_float_atb import router_py_templates as templates

    weight_names = parsed_model.get('weight_names', {})
    model_name_lower = weight_names.get('model_name', '')
    model_name_capital = model_name_lower.capitalize()

    rr = ""
    rr += templates.copyright.format(year=time.localtime().tm_year)
    rr += templates.import_formater.format()
    rr += templates.class_router_formater.format(model_name_capital=model_name_capital, pe_type=weight_names.get('pe_type'))

    save_name = utils.init_save_name(f"router_{model_name_lower}" if save_name is None else save_name) + ".py"
    save_dir = utils.init_save_dir(model_name_lower if save_dir is None else save_dir, sub_dir=".")
    save_path = os.path.join(save_dir, save_name)
    with open(save_path, "w") as ff:
        ff.write(rr)
    return save_path, rr