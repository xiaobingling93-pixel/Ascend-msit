import os
import time
from msit_llm.transform.torch_to_float_atb import utils
        
def match(keyword_lists, acl_inputs_name):
    for keyword_list in keyword_lists:
        for name in acl_inputs_name:
            if all(kw in name for kw in keyword_list):
                return name
    return 'PLACEHOLDER'


def flash_causal_py_gen(parsed_model, save_name=None, save_dir=None):
    from msit_llm.transform.torch_to_float_atb import flash_causal_py_templates as templates

    weight_names = parsed_model.get('weight_names', {})
    model_name_lower = weight_names.get('model_name', '')
    model_name_capital = model_name_lower.capitalize()

    rr = ""
    rr += templates.copyright.format(year=time.localtime().tm_year)
    rr += templates.import_formater.format(
        model_name_lower=model_name_lower,
        model_name_capital=model_name_capital,
    )

    acl_inputs_name = parsed_model.get('acl_inputs_name', [])

    acl_inputs_code_block = templates.acl_inputs_code_block.format(
        acl_inputs_name=str(acl_inputs_name),
        input_ids=match(["INPUT ID".split(), "INPUT"], acl_inputs_name),
        input_embedding=match(["INPUT EMBEDDING".split()], acl_inputs_name),
        position_ids=match(["POSITION ID".split()], acl_inputs_name),
        cos_embed=match(["COS".split()], acl_inputs_name),
        sin_embed=match(["SIN".split()], acl_inputs_name),
        atten_mask=match(["ATTENTION MASK".split()], acl_inputs_name),
        block_tables=match(["BLOCK".split()], acl_inputs_name),
        slots=match(["SLOTS".split()], acl_inputs_name),
        input_lengths=match(["INPUT LENGTH".split(), "SEQ LEN".split()], acl_inputs_name), 
        lm_head_indices=match(["LOGTIS INDICE".split(), "LOGITS INDICE".split()], acl_inputs_name), 
        q_lens=match(["Q LEN".split()], acl_inputs_name),
    )


    rr += templates.class_flash_causal_lm_formater.format(
        model_name_capital=model_name_capital, 
        model_prefix=weight_names.get('model_prefix'),   
        lmhead=weight_names.get('lmhead'),
        model_name_in_atb_framework=weight_names.get('model_name_in_atb_framework').replace('/', '_'),
        qkv_sep=weight_names.get('qkv_sep'),
        mlp_sep=weight_names.get('mlp_sep'),
        acl_inputs_code_block=acl_inputs_code_block,
    )

    save_name = utils.init_save_name(f"flash_causal_{model_name_lower}" if save_name is None else save_name) + ".py"
    save_dir = utils.init_save_dir(model_name_lower if save_dir is None else save_dir, sub_dir=".")
    save_path = os.path.join(save_dir, save_name)
    with open(save_path, "w") as ff:
        ff.write(rr)
    return save_path, rr