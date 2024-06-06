import os
import time
from ait_llm.transform.torch_to_float_atb import utils
from ait_llm.transform.model_parser import parser

def float_model_cpp_gen(model, save_name=None, save_dir=None):
    """
    >>> from ait_llm.transform.torch_to_float_atb import float_model_cpp_templates
    >>> from ait_llm.transform.torch_to_float_atb import float_model_cpp_gen
    >>> import transformers

    >>> cc = transformers.models.llama.LlamaConfig()
    >>> cc.num_hidden_layers = 4
    >>> mm = transformers.AutoModelForCausalLM.from_config(cc)
    >>> rr = float_model_cpp_gen.float_model_cpp_gen(mm)
    """
    from ait_llm.transform.torch_to_float_atb import float_model_cpp_templates  # avoiding circular import

    parsed_model = parser.build_model_tree(model)
    model_name_lower = parsed_model.get("name", "model").lower()

    rr = ""
    rr += float_model_cpp_templates.copyright_header.format(year=time.localtime().tm_year)
    rr += float_model_cpp_templates.include_header_formater.format(
        model_name_lower=model_name_lower,
        other_operations="",
    )

    pre_properties = "\n".join([
        float_model_cpp_templates.weight_count_formatter,
        float_model_cpp_templates.operation_count_formatter,
        float_model_cpp_templates.in_tensor_id_formatter,
        float_model_cpp_templates.internal_tensor_id_formatter,
        float_model_cpp_templates.out_tensor_id_formatter,
        float_model_cpp_templates.from_string_formatter,
    ])

    build_graph = float_model_cpp_templates.build_graph_formatter.format(
        build_graph_pre_process_formatter=float_model_cpp_templates.build_graph_pre_process_formatter,
        build_graph_pre_process_norm_formatter=float_model_cpp_templates.build_graph_pre_process_norm_formatter,
        build_graph_layers_formatter=float_model_cpp_templates.build_graph_layers_formatter,
        build_graph_post_process_norm_formatter=float_model_cpp_templates.build_graph_post_process_norm_formatter,
        build_graph_post_process_lmhead_formatter=float_model_cpp_templates.build_graph_post_process_lmhead_formatter,
    )

    post_properties = "\n".join([
        float_model_cpp_templates.infer_shape_formatter,
        build_graph,
        float_model_cpp_templates.parse_param_formatter,
        float_model_cpp_templates.bind_param_host_tensor_formatter,
    ])

    rr += float_model_cpp_templates.basic_class_formatter.format(
        model_name_lower=model_name_lower,
        pre_properties=pre_properties,
        post_properties=post_properties,
    )

    save_name = utils.init_save_name(save_name) + ".cpp"
    save_dir = utils.init_save_dir(model_name_lower if save_dir is None else save_dir, sub_dir="model")
    save_path = os.path.join(save_dir, save_file)
    with open(save_path, "w") as ff:
        ff.write(rr)
    return save_path, rr