import os
import time
from msit_llm.transform.torch_to_float_atb import utils

def modeling_py_gen(parsed_model, save_name=None, save_dir=None):
    from msit_llm.transform.torch_to_float_atb import modeling_py_templates as templates

    weight_names = parsed_model.get('weight_names', {})
    model_name_lower = weight_names.get('model_name', '')
    model_name_capital = model_name_lower.capitalize()

    rr = ""
    rr += templates.copyright.format(year=time.localtime().tm_year)
    rr += templates.import_formater.format(
        model_name_lower=model_name_lower,
        model_name_capital=model_name_capital,
    )
    
    mlp_sep = weight_names.get('mlp_sep', [])
    if len(mlp_sep) == 2:        
        mlp_code_block = templates.mlp_sep_formater.format(
            gate_up_proj=weight_names.get('gate_up_proj'),
            gate_proj=mlp_sep[0],
            up_proj=mlp_sep[1],
            post_attention_layernorm=weight_names.get('post_attention_layernorm'),
            mlp_bias=weight_names.get('mlp_bias'),
            down_proj=weight_names.get('down_proj'), 
        )
    else:
        mlp_code_block = templates.mlp_pack_formater.format(
            gate_up_proj=weight_names.get('gate_up_proj'),
            layer_prefix=weight_names.get('layer_prefix'),
            post_attention_layernorm=weight_names.get('post_attention_layernorm'),
            mlp_bias=weight_names.get('mlp_bias'),
            down_proj=weight_names.get('down_proj'), 
        )

    qkv_sep = weight_names.get('qkv_sep', [])
    if len(qkv_sep) == 3:        
        qkv_code_block = templates.qkv_sep_formater.format(
            q_proj=qkv_sep[0],
            k_proj=qkv_sep[1],
            v_proj=qkv_sep[2],
            query_key_value=weight_names.get('query_key_value'),
            query_key_value_bias=weight_names.get('query_key_value_bias'),
            input_layernorm=weight_names.get('input_layernorm'),
        )
    else:
        qkv_code_block = templates.qkv_pack_formater.format(
            query_key_value=weight_names.get('query_key_value'),
            query_key_value_bias=weight_names.get('query_key_value_bias'),
            input_layernorm=weight_names.get('input_layernorm'),
        )

    word_embeddings_layernorm = weight_names.get('word_embeddings_layernorm')
    if word_embeddings_layernorm is None:
        word_embeddings_layernorm_code_block = ''
    else:
        word_embeddings_layernorm_code_block = templates.word_embeddings_layernorm_formater.format(
            model_name_capital=model_name_capital, 
            word_embeddings_layernorm=word_embeddings_layernorm,
            RMSNormClass='RMSNormBias' if weight_names.get('layernorm_bias') else 'RMSNorm',        
            )


    rr += templates.class_flash_model_formater.format(
        model_name_capital=model_name_capital, 
        model_prefix=weight_names.get('model_prefix'),
        layers_prefix=weight_names.get('layers_prefix'),     
        mlp_code_block=mlp_code_block,
        qkv_code_block=qkv_code_block,
        o_proj=weight_names.get('o_proj'),
        o_proj_bias=weight_names.get('o_proj_bias'),
        mlp=weight_names.get('mlp'),
        input_layernorm=weight_names.get('input_layernorm'),
        post_attention_layernorm=weight_names.get('post_attention_layernorm'),
        word_embeddings_layernorm_code_block=word_embeddings_layernorm_code_block,
        RMSNormClass='RMSNormBias' if weight_names.get('layernorm_bias') else 'RMSNorm',
        layernorm=weight_names.get('layernorm'),
        word_embeddings=weight_names.get('word_embeddings'),
        self_attention=weight_names.get('self_attention'),
    )

    save_name = utils.init_save_name(f"modeling_{model_name_lower}" if save_name is None else save_name) + ".py"
    save_dir = utils.init_save_dir(model_name_lower if save_dir is None else save_dir, sub_dir=".")
    save_path = os.path.join(save_dir, save_name)
    with open(save_path, "w") as ff:
        ff.write(rr)
    return save_path, rr