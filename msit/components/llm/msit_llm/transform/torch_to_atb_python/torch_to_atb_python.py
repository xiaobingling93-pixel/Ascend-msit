import os
import sys
import json
import re
import json
import inspect
from copy import deepcopy
from collections import namedtuple

from msit_llm.common.log import logger, set_log_level
set_log_level('debug')

CONFIG_ATTR_CANDIDATES = {
    "num_hidden_layers": ['num_hidden_layers', 'num_layers', 'n_layers'],
    "num_attention_heads": ['num_attention_heads'],
    "hidden_size": ['hidden_size'],
}

NN_MODULE_STACK = "nn_module_stack"
SKIP_NODES = ["size", "getitem", "to", "float", "finfo"]
TORCH_MODULE_TO_ATB_MAP = {
    "Embedding": dict(op_type="Gather", op_param=json.dumps({}), is_weights_first=True),
    "Gather": dict(op_type="Gather", op_param=json.dumps({})),
    ".*RMSNorm$": dict(op_type="RmsNorm", op_param=json.dumps({"layerType": "RMS_NORM_NORM"})),
    "Linear": dict(op_type="Linear", op_param=json.dumps({"hasBias": False})),
    ".*Rotary.*": dict(op_type="Rope", op_param=json.dumps({'rotaryCoeff':2})),
    ".*Attention$": dict(op_type="SelfAttention", op_param=json.dumps({'headNum': 32, 'kvHeadNum': 32, 'calcType':'PA_ENCODER'})),
    "SiLU": dict(op_type="Activation", op_param=json.dumps({'activationType':'ACTIVATION_SWISH'})),
    "add": dict(op_type="Elewise", op_param=json.dumps({'elewiseType':'ELEWISE_ADD'})),
    "mul": dict(op_type="Elewise", op_param=json.dumps({'elewiseType':'ELEWISE_MUL'})),
}
_FX_OP_TYPES = ['call_method', 'call_module', 'call_function', 'placeholder', 'output', 'get_attr']
FX_OP_TYPES = namedtuple('fx_op_types', _FX_OP_TYPES)(*_FX_OP_TYPES)


def get_config_attr(config, attr):
    if attr not in CONFIG_ATTR_CANDIDATES:
        return getattr(config, None)

    for sub_attr in CONFIG_ATTR_CANDIDATES[attr]:
        if hasattr(config, sub_attr):
            return getattr(config, sub_attr)
    return None

def build_model(source_path):
    try:
        from transformers import AutoConfig, AutoModelForCausalLM
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError("transformers package not found, try pip install transformers") from error

    try:
        config = AutoConfig.from_pretrained(source_path, trust_remote_code=True)
        config = try_setting_small_model(config)
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    except Exception as error:
        raise ValueError(f"build model from {source_path} failed, make sure it works within transformers") from error
    return model, config

def to_transformers_traced_module(model, input_names=("input_ids", "position_ids")):
    from transformers.utils.fx import symbolic_trace

    return symbolic_trace(mm, input_names=input_names)

def get_llama_source_code(function):
    return inspect.getsource(function).split('function=')[-1].split(', inputs=')[0].strip()

class Operation:
    def __init__(self, op_type, op_param="{}", inputs=[], outputs=[], op_name="", function=None, is_weights_first=False):
        self.op_type, self.op_param, self.inputs, self.outputs = op_type, op_param, inputs, outputs
        self.op_name, self.function, self.is_weights_first = op_name, function, is_weights_first

    def to_dict(self):
        return dict(
            op_type=self.op_type,
            op_param=self.op_param,
            inputs=self.inputs,
            outputs=self.outputs,
            op_name=self.op_name,
            function=self.function,
            is_weights_first=self.is_weights_first,
        )

    def to_json(self):
        return dict(
            op_type=self.op_type,
            op_param=self.op_param,
            inputs=self.inputs,
            outputs=self.op_name,
            op_name=self.outputs,
            function=self.function and get_llama_source_code(self.function),
            is_weights_first=self.is_weights_first,
        )

    def copy(self):
        return Operation(**deepcopy(self.to_dict()))


class TorchToATB:
    def __init__(self, torch_model, config=None, input_names=("input_ids", "position_ids")):
        self.torch_model, self.config = torch_model, config
        self.config = getattr(torch_model, "config", None) if config is None else config
        if self.config is None:
            logger.error("Invalid config")
            raise

        self.num_attention_heads = get_config_attr(config, "num_attention_heads")
        if self.num_attention_heads is None:
            logger.error("Invalid config")
            raise

        self.hidden_size = get_config_attr(config, "hidden_size")
        if self.hidden_size is None:
            logger.error("Invalid config")
            raise

        self.head_dim = self.hidden_size // self.num_attention_heads
        self.traced_module = to_transformers_traced_module(torch_model, input_names=input_names)
        self.model_name = torch_model.__class__.__name__

        self.weight_names, self.weight_stack_map = list(self.traced_module.state_dict().keys()), {}
        for ii in self.weight_names:
            self.weight_stack_map.setdefault('.'.join(ii.split('.')[:-1]), []).append(ii)

        self.torch_module_to_atb_map = {}
        for kk, vv in TORCH_MODULE_TO_ATB_MAP.items():
            re_key = re.compile(kk)
            if vv["op_type"] == "SelfAttention":
                op_param = json.loads(vv["op_param"])
                op_param["headNum"] = op_param["kvHeadNum"] = self.num_attention_heads
                vv["op_param"] = json.dumps(op_param)
            self.torch_module_to_atb_map[re_key] = Operation(**vv)

        self.pre_query_name, self.pre_key_name, self.pre_value_name, self.is_apply_rope = "", "", "", False
        self.model_inputs, self.model_outputs, self.operations = [], [], []

    @staticmethod
    def get_num_hidden_layers(config):
        candidates = ['num_hidden_layers', 'num_layers', 'n_layers']
        for attr in candidates:
            if hasattr(config, attr):
                return getattr(config, attr)
        return None

    @staticmethod
    def get_module_type_by_nn_module_stack(node):
        node_module_stack = list(node.meta[NN_MODULE_STACK].values())
        return None if len(node_module_stack) == 0 else node_module_stack[-1].__name__

    @staticmethod
    def get_module_name_by_nn_module_stack(node):
        node_module_stack = list(node.meta[NN_MODULE_STACK].keys())
        return None if len(node_module_stack) == 0 else node_module_stack[-1]

    @staticmethod
    def should_skip_node(node):
        if node.op == FX_OP_TYPES.call_method and node.target in SKIP_NODES:
            return True
        if node.op == FX_OP_TYPES.call_function and node.target.__name__ in SKIP_NODES:
            return True
        if node.op == FX_OP_TYPES.call_function and node.meta.get("is_wrapped", False):
            return True
        return False


    def find_in_torch_module_to_atb_map(self, node_type):
        for kk, vv in self.torch_module_to_atb_map.items():
            if kk.fullmatch(node_type):
                return vv.copy()
        return None

    def convert_module(self, node_module_type, node_module_name, input_names):
        atb_operation = self.find_in_torch_module_to_atb_map(node_module_type)
        outputs = [node_module_name + ".out"]
        atb_operation.op_name = node_module_name
        atb_operation.inputs = getattr(atb_operation, "inputs", []) + input_names
        atb_operation.outputs = getattr(atb_operation, "outputs", []) + outputs
        return atb_operation

    def get_node_type_and_inputs_and_name(self, node, output_node_map={}):
        cur_module_name = self.get_module_name_by_nn_module_stack(node)
        if node.op == FX_OP_TYPES.call_function and self.find_in_torch_module_to_atb_map(node.target.__name__):
            node_module_type = node.target.__name__
            # No other inputs if function
            cur_inputs = []
            for ii in node.all_input_nodes:
                if ii.name not in output_node_map:
                    continue
                cur_inputs += output_node_map[ii.name]
            module_name = cur_module_name + "." + node.name
        else:
            node_module_type = self.get_module_type_by_nn_module_stack(node)
            # None marks for placeholder of other inputs
            cur_inputs = self.weight_stack_map.get(cur_module_name, []) + [None]
            module_name = cur_module_name
        return node_module_type, cur_inputs, module_name

    def check_and_set_pre_qkv_name(self, atb_operation):
        if atb_operation.op_type == "Linear":
            sub_name = atb_operation.op_name.split(".")[-1]
            if "q" in sub_name:
                self.pre_query_name = atb_operation.outputs[0]
            elif "k" in sub_name:
                self.pre_key_name = atb_operation.outputs[0]
            elif "v" in sub_name:
                self.pre_value_name = atb_operation.outputs[0]

    def op_process_rope(self, atb_operation=None, module_name=""):
        self.is_apply_rope = True
        self.model_inputs += ['cos_table', 'sin_table']
        self.operations.append(self.convert_module("Gather", "gather_cos", ['cos_table', "position_ids"]))  # [TODO] Second inputs?
        self.operations.append(self.convert_module("Gather", "gather_sin", ['sin_table', "position_ids"]))  # [TODO] Second inputs?

    def op_process_attention(self, atb_operation=None, module_name=""):
        atb_operation.inputs = [module_name + '.q_embed_', module_name + '.k_embed_', module_name + '.v_embed_', "seq_len"]
        outputs = [ii + "_" for ii in atb_operation.outputs]

        if self.is_apply_rope:
            inputs = [self.pre_query_name, self.pre_key_name, "gather_cos.out", "gather_sin.out", "seq_len"]
            outputs = [module_name + '.q_embed', module_name + '.k_embed']
            self.operations.append(Operation(
                op_type="Rope",
                op_param=json.dumps({'rotaryCoeff':2}),
                inputs=inputs,
                outputs=outputs,
                op_name=module_name + ".rope",
            ))

        num_attention_heads, head_dim = self.num_attention_heads, self.head_dim  # Essential for `to_file`
        self.operations += [
            Operation(
                op_type="add_reshape",
                function=lambda org_shape: [org_shape[0], num_attention_heads, head_dim],
                inputs=[module_name + ".q_embed"],
                outputs=[module_name + '.q_embed_'],
                op_name="",
            ),
            Operation(
                op_type="add_reshape",
                function=lambda org_shape: [org_shape[0], num_attention_heads, head_dim],
                inputs=[module_name + ".k_embed"],
                outputs=[module_name + '.k_embed_'],
                op_name="",
            ),
            Operation(
                op_type="add_reshape",
                function=lambda org_shape: [org_shape[0], num_attention_heads, head_dim],
                inputs=[self.pre_value_name],
                outputs=[module_name + '.v_embed_'],
                op_name="",
            ),
            Operation(
                op_type="ReshapeAndCache",
                op_param=json.dumps({}),
                inputs=[module_name + '.k_embed_', module_name + '.v_embed_', 'k_cache', 'v_cache', 'slots_mapping'],
                outputs=['k_cache', 'v_cache'],
                op_name=module_name + ".reshape_and_cache",
            ),
            atb_operation,
            Operation(
                op_type="add_reshape",
                function=lambda org_shape: [org_shape[0], org_shape[1] * org_shape[2]],
                inputs=atb_operation.outputs,
                outputs=outputs,
                op_name=""
            ),
        ]

        if "k_cache" not in self.model_inputs:
            self.model_inputs += ['k_cache', 'v_cache', 'slots_mapping', 'seq_len']

    def refine_inputs_outputs(self, input_node_map, output_node_map, operation_outputs):
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

        for op in self.operations:
            if not isinstance(op, list) and (len(op.inputs) == 0 or op.inputs[-1] is not None):
                continue

            op.inputs = op.inputs[:-1]  # Exclude last None
            if op.is_weights_first:
                op.inputs = op.inputs + gathered_module_inputs.get(op.op_name, [])
            else:
                op.inputs = gathered_module_inputs.get(op.op_name, []) + op.inputs

    def convert_fx_traced_module(self):
        self.model_inputs, self.model_outputs, self.operations, self.is_apply_rope = [], [], [], False
        previous_module_name, cur_module_name, previous_operation_out, base_module_name = None, None, None, None
        input_node_map, output_node_map, operation_outputs = {}, {}, {}

        for node in self.traced_module.graph.nodes:
            logger.debug(f"\nnode.name = {node.name}, node.op = {node.op}")
            if node.op == FX_OP_TYPES.placeholder:  # Input node
                self.model_inputs.append(node.name)
                input_node_map[node.name] = [node.name]
                output_node_map[node.name] = [node.name]
                continue
            if node.op == FX_OP_TYPES.output:  # Output node
                self.model_outputs.append(node.name)
                continue
            if not hasattr(node, "meta") or not node.meta.get(NN_MODULE_STACK, []):
                continue

            if base_module_name is None:
                base_module_name = list(node.meta[NN_MODULE_STACK].keys())[0]

            cur_module_name = self.get_module_name_by_nn_module_stack(node)
            input_node_map.setdefault(cur_module_name, []).extend([ii.name for ii in node.all_input_nodes])

            logger.debug(f"cur_module_name = {cur_module_name}, previous_module_name = {previous_module_name}")
            if cur_module_name != base_module_name:
                output_node_map[node.name] = previous_operation_out  # will be overwrited later
            if cur_module_name == previous_module_name:
                continue
            if self.should_skip_node(node):
                continue
            previous_module_name = cur_module_name

            logger.debug(f"cur_module_name = {cur_module_name}, node.name = {node.name}")
            node_module_type, cur_inputs, module_name = self.get_node_type_and_inputs_and_name(node, output_node_map)
            if self.find_in_torch_module_to_atb_map(node_module_type) is None:
                logger.warning(f"node not supported: node.name = {node.name}, node.type = {node_module_type}")
                continue
            if len(cur_inputs) == 0:
                logger.warning(f"found none valid input: node.name = {node.name}, node.type = {node_module_type}")
                continue

            atb_operation = self.convert_module(node_module_type, module_name, cur_inputs)
            self.check_and_set_pre_qkv_name(atb_operation)
            logger.debug(f"atb_operation = {atb_operation.to_json()}")
            if not self.is_apply_rope and atb_operation.op_type == "Rope":
                self.op_process_rope(atb_operation=atb_operation, module_name=module_name)
            elif atb_operation.op_type == "SelfAttention":
                self.op_process_attention(atb_operation=atb_operation, module_name=module_name)
                cur_outputs = self.operations[-1].outputs
                output_node_map[node.name] = previous_operation_out = operation_outputs[cur_module_name] = cur_outputs
            else:
                self.operations.append(atb_operation)
                operation_outputs[cur_module_name] = atb_operation.outputs
                output_node_map[node.name] = previous_operation_out = atb_operation.outputs

        self.refine_inputs_outputs(input_node_map, output_node_map, operation_outputs)
        self.operations[-1].outputs = self.model_outputs
        return self.model_inputs, self.model_outputs, self.weight_names, self.operations

    def build_atb_model(self):
        import torch
        import torch_npu

        path = os.getenv('ATB_SPEED_HOME_PATH')
        sys.path.append(os.path.join(path, 'lib'))
        import _libatb_torch as atb

        atb_model = atb._GraphOperation(self.model_name)
        in_tensors = self.model_inputs + self.weight_names
        operation_inputs = [jj for ii in self.operations for jj in ii.inputs]
        valid_inputs = [ii for ii in in_tensors if ii in operation_inputs]
        atb_model.add_input_output(input=valid_inputs, output=self.operations[-1].outputs)
        for id, op in enumerate(self.operations):
            if op.op_type == "add_reshape":
                atb_model.add_reshape(op.inputs[0], op.outputs[0], op.function)
            else:
                atb_model.add_operation(
                    operation=atb._BaseOperation(op_type=op.op_type, op_param=op.op_param, op_name=op.op_name),
                    input=op.inputs,
                    output=op.outputs,
                )
        atb_model.build()
        return atb_model

    def to_file(self, output_path):
        in_tensors = self.model_inputs + self.weight_names
        operation_inputs = [jj for op in self.operations for jj in op.inputs]
        valid_inputs = [ii for ii in in_tensors if ii in operation_inputs]
        intensors_str = '[\n' + ''.join(["    '{}',\n".format(ii) for ii in valid_inputs]) + ']'

        contents = "\n".join([
            "import os",
            "import sys",
            "import json",
            "import torch",
            "import torch_npu",
            "torch_npu.npu.set_device(0)",
            "",
            "path = os.getenv('ATB_SPEED_HOME_PATH')",
            "sys.path.append(os.path.join(path, 'lib'))",
            "import _libatb_torch as atb",
            "",
            f'num_attention_heads, head_dim = {self.num_attention_heads}, {self.head_dim}',
            f"aa = atb._GraphOperation('{self.model_name}')",
            f"in_tensors = {intensors_str}",
            f"aa.add_input_output(input=in_tensors, output={self.operations[-1].outputs})",
            "",
        ]) + "\n"

        for op in self.operations:
            if op.op_type == "add_reshape":
                function = get_llama_source_code(op.function)
                contents += f"""aa.add_reshape('{op.inputs[0]}', '{op.outputs[0]}', {function})\n"""
            else:
                contents += "aa.add_operation(\n    "
                contents += ",\n    ".join([
                    f"operation=atb._BaseOperation(op_type='{op.op_type}', op_param='{op.op_param}', op_name='{op.op_name}')",
                    f"input={op.inputs}",
                    f"output={op.outputs}",
                ])
                contents += ",\n)\n"
        contents += "aa.build()"

        with open(output_path, 'w') as ff:
            ff.write(contents)
        return contents