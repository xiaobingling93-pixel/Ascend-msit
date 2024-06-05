import os
import time
from ait_llm.transform.torch_to_float_atb import float_model_h_templates
from ait_llm.transform.model_parser import parser

def float_model_h_gen(model, save_file=None, save_dir=None):
    """
    >>> from ait_llm.transform.torch_to_float_atb import float_model_h_templates
    >>> from ait_llm.transform.torch_to_float_atb import float_model_h_gen
    >>> import transformers

    >>> cc = transformers.models.llama.LlamaConfig()
    >>> cc.num_hidden_layers = 4
    >>> mm = transformers.AutoModelForCausalLM.from_config(cc)
    >>> rr = float_model_h_gen.float_model_h_gen(mm)
    """
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

    save_file = "decoder_model.h" if save_file is None else save_file
    save_dir = os.path.join(model_name_lower, "model") if save_dir is None else save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, save_file)
    with open(save_path, "w") as ff:
        ff.write(rr)
    print("Saved:", save_path)
    return rr