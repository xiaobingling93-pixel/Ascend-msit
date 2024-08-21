import json

# [TODO] gt, ne, repeat, arange, model_causal_mask, full, expand, "view", "transpose", "unsqueeze", "mul", "floordiv", "neg", "cat"
SKIP_NODES = ["size", "getitem", "to", "float", "finfo"]
TORCH_MODULE_TO_ATB_MAP = {
    "Embedding": dict(op_type="Gather", op_param=json.dumps({}), is_weights_first=False),
    "Gather": dict(op_type="Gather", op_param=json.dumps({})),
    "LlamaRMSNorm": dict(op_type="RmsNorm", op_param=json.dumps({"layerType": "RMS_NORM_NORM"})),
    "Linear": dict(op_type="Linear", op_param=json.dumps({"hasBias": False})),
    "LlamaRotaryEmbedding": dict(op_type="Rope", op_param=json.dumps({'rotaryCoeff':2})),
    "LlamaAttention": dict(op_type="SelfAttention", op_param=json.dumps({'headNum': 32, 'kvHeadNum': 32, 'calcType':'PA_ENCODER'})),
    "SiLU": dict(op_type="Activation", op_param=json.dumps({'activationType':'ACTIVATION_SWISH'})),
    "add": dict(op_type="Elewise", op_param=json.dumps({'elewiseType':'ELEWISE_ADD'})),
    "mul": dict(op_type="Elewise", op_param=json.dumps({'elewiseType':'ELEWISE_MUL'})),
}


def get_module_type_by_nn_module_stack(node):
    node_module_stack = list(node.meta["nn_module_stack"].values())
    return None if len(node_module_stack) == 0 else node_module_stack[-1].__name__


def get_module_name_by_nn_module_stack(node, is_in_first_repeat_block=True):
    node_module_stack = list(node.meta["nn_module_stack"].keys())
    return None if len(node_module_stack) == 0 else node_module_stack[-1]


def convert_module(node_module_type, node_module_name, input_names):
    atb_operation = TORCH_MODULE_TO_ATB_MAP[node_module_type].copy()
    outputs = [node_module_name + ".out"]
    atb_operation["op_name"] = node_module_name
    atb_operation["inputs"] = atb_operation.get("inputs", []) + input_names
    atb_operation["outputs"] = atb_operation.get("outputs", []) + outputs
    return atb_operation


def convert_fx_traced_module_wo_repeat(traced_module):
    # [TODO] get from model
    head_num, head_dim = 32, 128

    weight_names, weight_stack_map = list(traced_module.state_dict().keys()), {}
    for ii in weight_names:
        weight_stack_map.setdefault('.'.join(ii.split('.')[:-1]), []).append(ii)

    model_inputs, model_outputs, operations, operation_outputs = [], [], [], {}
    previous_module_name, cur_module_name, previous_operation_out, is_apply_rope = None, None, None, False
    cur_module_inputs, input_node_map, output_node_map = [], {}, {}
    pre_query_name, pre_key_name, pre_value_name = "", "", ""

    for node in traced_module.graph.nodes:
        # print(f"\n>>>> {node.name = }, {node.op = }")
        if node.op == "placeholder":  # Input node
            model_inputs.append(node.name)
            input_node_map[node.name] = [node.name]
            output_node_map[node.name] = [node.name]
            continue
        if node.op == "output":  # Output node
            model_outputs.append(node.name)
            continue
        if not hasattr(node, "meta") or "nn_module_stack" not in node.meta:
            continue

        cur_module_name = get_module_name_by_nn_module_stack(node, is_in_first_repeat_block=False)
        # print(f">>>> {cur_module_name = }, {previous_module_name = }")
        input_node_map.setdefault(cur_module_name, []).extend([ii.name for ii in node.all_input_nodes])
        if cur_module_name != "model":  # [TODO] actual model name
            output_node_map[node.name] = previous_operation_out  # will be overwrited later
        if cur_module_name == previous_module_name:
            continue

        if node.op == "call_method" and node.target in SKIP_NODES:
            continue
        if node.op == "call_function" and node.target.__name__ in SKIP_NODES:
            continue
        if node.op == "call_function" and node.meta.get("is_wrapped", False):
            continue
        previous_module_name = cur_module_name

        # print(f">>>> {cur_module_name = } {node.name = }")
        if node.op == 'call_function' and node.target.__name__ in TORCH_MODULE_TO_ATB_MAP:
            node_module_type = node.target.__name__
            # No other inputs if function
            cur_inputs = []
            for ii in node.all_input_nodes:
                if ii.name not in output_node_map:
                    continue
                # cur_input_name = output_node_map[ii.name]
                # cur_inputs += operation_outputs.get(cur_input_name, [cur_input_name])
                cur_inputs += output_node_map[ii.name]
            module_name = cur_module_name + "." + node.name
        else:
            node_module_type = get_module_type_by_nn_module_stack(node)
            cur_inputs = weight_stack_map.get(cur_module_name, []) + [None]  # None marks for placeholder of other inputs
            module_name = cur_module_name
        if node_module_type not in TORCH_MODULE_TO_ATB_MAP:
            print(f">>>> Not suppoorted: {node.name = }")
            continue
        if len(cur_inputs) == 0:
            print(f">>>> found none valid inputs")
            continue

        atb_operation = convert_module(node_module_type, module_name, cur_inputs)
        if atb_operation["op_type"] == "Linear":
            sub_name = atb_operation["op_name"].split(".")[-1]
            if "q" in sub_name:
                pre_query_name = atb_operation["outputs"][0]
            elif "k" in sub_name:
                pre_key_name = atb_operation["outputs"][0]
            elif "v" in sub_name:
                pre_value_name = atb_operation["outputs"][0]

        if not is_apply_rope and atb_operation["op_type"] == "Rope":
            is_apply_rope = True
            model_inputs += ['cos_table', 'sin_table']
            operations.append(convert_module("Gather", "gather_cos", ['cos_table', "position_ids"]))  # [TODO] Second inputs?
            operations.append(convert_module("Gather", "gather_sin", ['sin_table', "position_ids"]))  # [TODO] Second inputs?
        elif atb_operation["op_type"] == "SelfAttention":
            atb_operation["inputs"] = [module_name + '.q_embed_', module_name + '.k_embed_', module_name + '.v_embed_', "seq_len"]
            outputs = [ii + "_" for ii in atb_operation["outputs"]]

            if is_apply_rope:
                operations.append(dict(op_type="Rope", op_param=json.dumps({'rotaryCoeff':2}), inputs=[pre_query_name, pre_key_name, "gather_cos.out", "gather_sin.out", "seq_len"], outputs=[module_name + '.q_embed', module_name + '.k_embed'], op_name=module_name + ".rope")),
            operations += [
                dict(op_type="add_reshape", function=lambda org_shape: [org_shape[0], head_num, head_dim], inputs=[module_name + ".q_embed"], outputs=[module_name + '.q_embed_'], op_name=""),
                dict(op_type="add_reshape", function=lambda org_shape: [org_shape[0], head_num, head_dim], inputs=[module_name + ".k_embed"], outputs=[module_name + '.k_embed_'], op_name=""),
                dict(op_type="add_reshape", function=lambda org_shape: [org_shape[0], head_num, head_dim], inputs=[pre_value_name], outputs=[module_name + '.v_embed_'], op_name=""),
                dict(op_type="ReshapeAndCache", op_param=json.dumps({}), inputs=[module_name + '.k_embed_', module_name + '.v_embed_', 'k_cache', 'v_cache', 'slots_mapping'], outputs=['k_cache', 'v_cache'], op_name=module_name + ".reshape_and_cache"),
                atb_operation,
                dict(op_type="add_reshape", function=lambda org_shape: [org_shape[0], org_shape[1] * org_shape[2]], inputs=atb_operation["outputs"], outputs=outputs, op_name=""),
            ]
            operation_outputs[cur_module_name] = outputs
            output_node_map[node.name] = previous_operation_out = outputs
            if "k_cache" not in model_inputs:
                model_inputs += ['k_cache', 'v_cache', 'slots_mapping', 'seq_len']
        else:
            operations.append(atb_operation)
            operation_outputs[cur_module_name] = atb_operation["outputs"]
            output_node_map[node.name] = previous_operation_out = atb_operation["outputs"]

    gathered_module_inputs = {}
    for kk, vv in input_node_map.items():
        gathered_inputs = []
        for ii in vv:
            if ii in output_node_map: # and output_node_map[ii] != kk and output_node_map[ii] != kk + ".out":
                gathered_inputs += output_node_map[ii]
        for ii in set(gathered_inputs):
            if ii == kk or ii == kk + ".out":
                continue
            gathered_module_inputs.setdefault(kk, []).extend(operation_outputs.get(ii, ii if isinstance(ii, list) else [ii]))

    for op in operations:
        if not isinstance(op, list) and (len(op["inputs"]) == 0 or op["inputs"][-1] is not None):
            continue

        op['inputs'] = op['inputs'][:-1]  # Exclude last None
        if op.pop("is_weights_first", True):
            op['inputs'] = op['inputs'] + gathered_module_inputs.get(op["op_name"], [])
        else:
            op['inputs'] = gathered_module_inputs.get(op["op_name"], []) + op['inputs']

    return model_inputs, model_outputs, weight_names, operations


def build_atb_model_wo_repeat_str(model_name, model_inputs, model_outputs, weight_names, operations):
    in_tensors = model_inputs + weight_names
    operation_inputs = [jj for ii in operations for jj in ii['inputs']]
    valid_inputs = [ii for ii in in_tensors if ii in operation_inputs]
    intensors_str = '[\n' + ''.join(["    '{}',\n".format(ii) for ii in valid_inputs]) + ']'

    contents = f"""
    import os
    import sys
    import json
    import torch
    import torch_npu
    torch_npu.npu.set_device(0)

    path = os.getenv('ATB_SPEED_HOME_PATH')
    sys.path.append(os.path.join(path, 'lib'))
    import _libatb_torch as atb

    aa = atb._GraphOperation('{model_name}')
    in_tensors = {intensors_str}
    aa.add_input_output(input=in_tensors, output={operations[-1]["outputs"]})
    """

    for op in operations:
        if op["op_type"] == "add_reshape":
            function = inspect.getsource(op["function"]).split('function=')[-1].split(', inputs=')[0]
            contents += f"""aa.add_reshape('{op["inputs"][0]}', '{op["outputs"][0]}', {function})\n"""
        else:
            contents += "aa.add_operation(\n    "
            contents += ",\n    ".join([
                f"operation=atb._BaseOperation(op_type='{op['op_type']}', op_param='{op['op_param']}', op_name='{op['op_name']}')",
                f"input={op['inputs']}",
                f"output={op['outputs']}",
            ])
            contents += ",\n)\n"
    contents += """
    aa.build()
    """
    return contents

def build_atb_model_wo_repeat(model_name, model_inputs, model_outputs, weight_names, operations, operations_slice=None):
    import os
    import sys
    import json
    import torch
    import torch_npu
    torch_npu.npu.set_device(0)

    path = os.getenv('ATB_SPEED_HOME_PATH')
    sys.path.append(os.path.join(path, 'lib'))
    import _libatb_torch as atb

    sub_operations = operations[:operations_slice]
    aa = atb._GraphOperation(model_name)
    in_tensors = model_inputs + weight_names
    operation_inputs = [jj for ii in sub_operations for jj in ii['inputs']]
    valid_inputs = [ii for ii in in_tensors if ii in operation_inputs]
    aa.add_input_output(input=valid_inputs, output=sub_operations[-1]["outputs"])
    for id, op in enumerate(sub_operations):
        # if id in [4, 8]:
        #     continue
        # print(id, op)
        # aa = atb._GraphOperation(model_name)
        if op["op_type"] == "add_reshape":
            aa.add_reshape(op["inputs"][0], op["outputs"][0], op["function"])
        else:
            # aa.add_input_output(input=op["inputs"], output=op["outputs"])
            aa.add_operation(
                operation=atb._BaseOperation(op_type=op["op_type"], op_param=op["op_param"], op_name=op["op_name"]),
                input=op["inputs"],
                output=op["outputs"],
            )
        # aa.build()
    aa.build()
    return aa

    hn = head_num = 32
    hd = head_dim = 128
    b = 1
    s = 512
    h = hn * hd
    max_s = 1024
    bn = 1024
    bs = 128
    layer_num = 30
    vocab_size = 12800
    width = 0.2

    wws = set(traced_module.state_dict().keys()) & set(aa.input_names)
    other_inputs = set(aa.input_names) - wws  # {'v_cache', 'k_cache', 'cos_table', 'sin_table', 'input_ids', 'position_ids', 'slots_mapping', 'seq_len'}
    llama_model_inputs = {kk: traced_module.state_dict()[kk].half().npu() for kk in wws}
    # if "input_ids" in other_inputs:

    llama_model_inputs.update(dict(
        input_ids=torch.arange(s).npu(),
        position_ids=torch.arange(s).npu(),
        cos_table=torch.rand(max_s, hd).half().npu() * width - width / 2,
        sin_table=torch.rand(max_s, hd).half().npu() * width - width / 2,
        k_cache=torch.zeros(bn, bs, hn, hd).half().npu(),
        v_cache=torch.zeros(bn, bs, hn, hd).half().npu(),
        slots_mapping=torch.zeros(b * s, dtype=torch.int).npu(),
        seq_len=(torch.ones(b, dtype=torch.int) * s).npu(),
    ))

    llama_model_outputs = {}
    llama_model_outputs[aa.output_names[0]] = torch.ones(b * s, vocab_size).half().npu()

    bind_map = {}
    bind_map['seq_len'] = llama_model_inputs["seq_len"].cpu()
    aa.forward(llama_model_inputs, llama_model_outputs, bind_map)


if __name__ == "__main__":
    import os

    llama_config = """{
      "architectures": [
        "LlamaForCausalLM"
      ],
      "bos_token_id": 1,
      "eos_token_id": 2,
      "hidden_act": "silu",
      "hidden_size": 1024,
      "initializer_range": 0.02,
      "intermediate_size": 4096,
      "max_position_embeddings": 4096,
      "model_type": "llama",
      "num_attention_heads": 4,
      "num_hidden_layers": 4,
      "num_key_value_heads": 4,
      "pad_token_id": 0,
      "pretraining_tp": 1,
      "rms_norm_eps": 1e-05,
      "rope_scaling": null,
      "tie_word_embeddings": false,
      "torch_dtype": "float16",
      "transformers_version": "4.31.0.dev0",
      "use_cache": true,
      "vocab_size": 32000
    }"""

    os.makedirs("test_transformers/", exist_ok=True)
    with open("test_transformers/config.json", "w") as ff:
        ff.write(llama_config)

    from transformers import AutoConfig, AutoModelForCausalLM

    cc = AutoConfig.from_pretrained("test_transformers/")
    cc.num_hidden_layers = 2
    mm = AutoModelForCausalLM.from_config(cc)
    model_name = mm.__class__.__name__

    from transformers.utils.fx import symbolic_trace

    traced_module = symbolic_trace(mm, input_names=["input_ids", "position_ids"])
    model_inputs, model_outputs, weight_names, operations = convert_fx_traced_module_wo_repeat(traced_module)
    print(f">>>> {model_inputs = }")
    print(f">>>> {model_outputs = }")
    print(">>>> weights:", json.dumps(weight_names, indent=2))
    # print(">>>> operations:", json.dumps(operations, indent=2))

    aa = build_atb_model_wo_repeat_str(model_name, model_inputs, model_outputs, weight_names, operations)


    llama_layer_inputs = {'inputs': torch.rand(512, 1024).half().npu(), 'qkv_weight': torch.rand(3 * 1024, 1024).half().npu()}
    llama_layer_outputs = {'layer_out': torch.ones(512, 3 * 1024).half().npu()}
    _ = aa.forward(llama_layer_inputs, llama_layer_outputs)

    atb_out = llama_layer_outputs['layer_out']
    torch_out = llama_layer_inputs['inputs'] @ llama_layer_inputs['qkv_weight'].T
    print("allclose:", torch.allclose(atb_out, torch_out))