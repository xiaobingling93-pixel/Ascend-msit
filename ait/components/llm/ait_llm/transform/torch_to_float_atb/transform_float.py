from ait_llm.common.log import logger

def transform_float(source_path, save_name=None, save_dir=None):
    from ait_llm.transform import torch_to_float_atb

    try:
        from transformers import AutoConfig, AutoModelForCausalLM
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError("transformers package not found, try pip install transformers") from error

    try:
        config = AutoConfig.from_pretrained(source_path, trust_remote_code=True)
        source_model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    except Exception as error:
        raise ValueError(f"build model from {source_path} failed, make sure it works within transformers") from error

    model_cpp_file, _ = torch_to_float_atb.float_model_cpp_gen(source_model, save_name=save_name, save_dir=save_dir)
    model_h_file, _ = torch_to_float_atb.float_model_h_gen(source_model, save_name=save_name, save_dir=save_dir)
    layer_cpp_file, _ = torch_to_float_atb.float_layer_cpp_gen(source_model, save_name=save_name, save_dir=save_dir)
    layer_h_file, _ = torch_to_float_atb.float_layer_h_gen(source_model, save_name=save_name, save_dir=save_dir)
    result_files = [model_cpp_file, model_h_file, layer_cpp_file, layer_h_file]
    print("Generated files: [\n    " + ",\n    ".join(result_files) + ",\n]")
    return result_files