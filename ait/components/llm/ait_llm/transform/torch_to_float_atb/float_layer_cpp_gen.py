import os
import time
from ait_llm.transform.torch_to_float_atb import float_layer_cpp_templates
from ait_llm.transform.model_parser import parser

def float_layer_gen(model, save_file=None, save_dir=None):
    """
    >>> from ait_llm.transform.torch_to_float_atb import float_layer_cpp_templates
    >>> from ait_llm.transform.torch_to_float_atb import float_layer_cpp_gen
    >>> import transformers

    >>> cc = transformers.models.llama.LlamaConfig()
    >>> cc.num_hidden_layers = 4
    >>> mm = transformers.AutoModelForCausalLM.from_config(cc)
    >>> rr = float_layer_cpp_gen.float_layer_gen(mm)
    """
    parsed_model = parser.build_model_tree(model)
    model_name_lower = parsed_model.get("name", "model").lower()

    rr = ""
    rr += float_layer_cpp_templates.cpp_copyright_header.format(year=time.localtime().tm_year) + "\n"
    rr += "\n".join([f'''#include "{i}"''' for i in float_layer_cpp_templates.all_atb_operation_headers]) + \
          f"""\n#include "models/{model_name_lower}/layer/decoder_layer.h""""

    layer_core_components = float_layer_cpp_templates.decoder_layer_formatter.format(
        attention_formatter=float_layer_cpp_templates.attention_formatter,
        residual_add_formatter=float_layer_cpp_templates.residual_add_formatter,
        mlp_formatter=float_layer_cpp_templates.mlp_formatter,
        mlp_residual_add_formatter=float_layer_cpp_templates.mlp_residual_add_formatter
    )

    post_properties = "\n".join([
        float_layer_cpp_templates.parse_param_formatter,
        float_layer_cpp_templates.bind_param_host_tensor_formatter,
    ])

    rr += float_layer_cpp_templates.basic_class_formatter.format(
        model_name_lower=model_name_lower,
        decoder_layer=layer_core_components,
        post_properties=post_properties,
    )

    save_file = "decoder_layer.cpp" if save_file is None else save_file
    save_dir = os.path.join(model_name_lower, "layer") if save_dir is None else save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, save_file)
    with open(save_path, "w") as ff:
        ff.write(rr)
    print("Saved:", save_path)
    return rr
