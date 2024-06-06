import os
import time
from ait_llm.transform.torch_to_float_atb import utils
from ait_llm.transform.model_parser import parser

def float_model_h_gen(model, save_name=None, save_dir=None):
    """
    >>> from ait_llm.transform.torch_to_float_atb import float_model_h_templates
    >>> from ait_llm.transform.torch_to_float_atb import float_model_h_gen
    >>> import transformers

    >>> cc = transformers.models.llama.LlamaConfig()
    >>> cc.num_hidden_layers = 4
    >>> mm = transformers.AutoModelForCausalLM.from_config(cc)
    >>> rr = float_model_h_gen.float_model_h_gen(mm)
    """
    from ait_llm.transform.torch_to_float_atb import float_model_h_templates  # avoiding circular import

    parsed_model = parser.build_model_tree(model)
    model_name_lower = parsed_model.get("name", "model").lower()

    rr = ""
    rr += float_model_h_templates.copyright_header.format(year=time.localtime().tm_year)
    rr += float_model_h_templates.include_header_formater.format(
        model_name_lower=model_name_lower,
        model_name_upper=model_name_lower.upper(),
    )

    rr += float_model_h_templates.basic_class_formatter.format(
        struct_param_formatter=float_model_h_templates.struct_param_formatter,
    )

    save_name = utils.init_save_name(save_name) + ".h"
    save_dir = utils.init_save_dir(model_name_lower if save_dir is None else save_dir, sub_dir="model")
    save_path = os.path.join(save_dir, save_file)
    with open(save_path, "w") as ff:
        ff.write(rr)
    return save_path, rr