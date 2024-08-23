# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import json
import re
import json
import inspect
from copy import deepcopy
from collections import namedtuple

from msit_llm.common.log import logger, set_log_level


CONFIG_ATTR_CANDIDATES = {
    "num_hidden_layers": ["num_hidden_layers", "num_layers", "n_layers"],
    "num_attention_heads": ["num_attention_heads"],
    "hidden_size": ["hidden_size"],
    "vocab_size": ["vocab_size"],
}

NN_MODULE_STACK = "nn_module_stack"
SKIP_NODES = ["size", "getitem", "to", "float", "finfo"]
TORCH_MODULE_TO_ATB_MAP = {
    "Embedding": dict(op_type="Gather", op_param={}, is_weights_first=True),
    "Gather": dict(op_type="Gather", op_param={}),
    ".*RMSNorm$": dict(op_type="RmsNorm", op_param={"layerType": "RMS_NORM_NORM"}),
    "Linear": dict(op_type="Linear", op_param={"hasBias": False}),
    ".*Rotary.*": dict(op_type="Rope", op_param={"rotaryCoeff": 2}),
    ".*Attention$": dict(op_type="SelfAttention", op_param={"headNum": 32, "kvHeadNum": 32, "calcType": "PA_ENCODER"}),
    "SiLU": dict(op_type="Activation", op_param={"activationType": "ACTIVATION_SWISH"}),
    "add": dict(op_type="Elewise", op_param={"elewiseType": "ELEWISE_ADD"}),
    "mul": dict(op_type="Elewise", op_param={"elewiseType": "ELEWISE_MUL"}),
}

_FX_OP_TYPES = ["call_method", "call_module", "call_function", "placeholder", "output", "get_attr"]
FX_OP_TYPES = namedtuple("FX_OP_TYPES", _FX_OP_TYPES)(*_FX_OP_TYPES)

_FIXED_INPUTS = {
    "input_ids",
    "position_ids",
    "cos_table",
    "sin_table",
    "k_cache",
    "v_cache",
    "slots_mapping",
    "seq_len",
}
FIXED_INPUTS = namedtuple("FIXED_INPUTS", _FIXED_INPUTS)(*_FIXED_INPUTS)
BASIC_INPUT_NAMES = (FIXED_INPUTS.input_ids, FIXED_INPUTS.position_ids)

_RESHPAE_KIND = ["reshape_qkv", "reshape_0_12"]
RESHPAE_KIND = namedtuple("RESHPAE_KIND", _RESHPAE_KIND)(*_RESHPAE_KIND)


def get_config_attr(config, attr, default=None):
    if attr not in CONFIG_ATTR_CANDIDATES:
        return getattr(config, attr, default)

    for sub_attr in CONFIG_ATTR_CANDIDATES[attr]:
        if hasattr(config, sub_attr):
            return getattr(config, sub_attr)
    return default


def build_model(source_path):
    try:
        from transformers import AutoConfig, AutoModelForCausalLM
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError("transformers package not found, try pip install transformers") from error

    try:
        config = AutoConfig.from_pretrained(source_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    except Exception as error:
        raise ValueError(f"build model from {source_path} failed, make sure it works within transformers") from error
    return model, config


def to_transformers_traced_module(model, input_names=BASIC_INPUT_NAMES):
    from transformers.utils.fx import symbolic_trace

    return symbolic_trace(model, input_names=input_names)


def get_lambda_source_code(function):
    source_code = inspect.getsource(function).split("function=")[-1].split(", inputs=")[0].strip()
    return source_code[:-1] if source_code.endswith(",") else source_code


class Operation:
    def __init__(self, op_type, op_param={}, inputs=[], outputs=[], op_name="", function=None, is_weights_first=False):
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
            op_param=json.dumps(self.op_param),
            inputs=self.inputs,
            outputs=self.op_name,
            op_name=self.outputs,
            function=self.function and get_lambda_source_code(self.function),
            is_weights_first=self.is_weights_first,
        )

    def copy(self):
        return Operation(**deepcopy(self.to_dict()))


class ATBModelConfig:
    def __init__(self, vocab_size, num_attention_heads, head_dim, max_batch_size=1, max_seq_len=1024, **kwargs):
        self.vocab_size, self.num_attention_heads, self.head_dim = vocab_size, num_attention_heads, head_dim
        self.max_batch_size, self.max_seq_len, self.kwargs = max_batch_size, max_seq_len, kwargs
        for kk, vv in kwargs.items():
            setattr(self, kk, vv)

    def to_dict(self):
        return dict(
            vocab_size=self.vocab_size,
            num_attention_heads=self.num_attention_heads,
            head_dim=self.head_dim,
            max_batch_size=self.max_batch_size,
            max_seq_len=self.max_seq_len,
            **self.kwargs
        )
    
    def __repr__(self):
        return self.to_dict()


class ATBModel:
    def __init__(self, atb_model, atb_model_config):
        import torch
        import torch_npu

        if isinstance(atb_model_config, dict):
            required_keys = ["vocab_size", "num_attention_heads", "head_dim"]
            if any([kk not in atb_model_config for kk in required_keys]):
                raise ValueError(f"atb_model_config is a dict but not containing all keys in {required_keys}")
            self.atb_model_config = ATBModelConfig(**atb_model_config)
        elif not isinstance(atb_model_config, ATBModelConfig):
            raise ValueError("atb_model_config is not an instance of ATBModelConfig")
        else:
            self.atb_model_config = atb_model_config

        self.atb_model = atb_model
        self.inputs = self.input_names = set(atb_model.input_names)
        self.outputs = self.output_names = atb_model.output_names
        self.weights, self.inv_freq_weight = {}, None
        self.torch = torch

        self.cache_shape = [
            self.atb_model_config.max_batch_size,
            self.atb_model_config.max_seq_len,
            self.atb_model_config.num_attention_heads,
            self.atb_model_config.head_dim,
        ]
        self.vocab_size = self.atb_model_config.vocab_size

    def set_weights(self, weights):
        source_weights = set(weights.keys())
        valid_weights = source_weights & self.inputs
        missing_weights = self.inputs - source_weights - set(FIXED_INPUTS)
        unused_weights = source_weights - self.inputs

        for kk in valid_weights:
            cur_weight = weights[kk]
            cur_weight = cur_weight.half() if str(cur_weight.dtype).split(".")[-1] == "float32" else cur_weight
            self.weights[kk] = cur_weight.npu()

        self.inv_freq_weight = None
        for weight_name in unused_weights:
            if "invfreq" in weight_name.split(".")[-1].replace("_", "").lower():
                self.inv_freq_weight = weights[weight_name].squeeze().half().npu()
                unused_weights.remove(weight_name)
                break

        if len(unused_weights) > 0:
            logger.warning(f"unused weights: {unused_weights}")
        if len(missing_weights) > 0:
            logger.warning(f"missing weights: {missing_weights}")

    def forward(self, input_ids, position_ids, k_cache=None, v_cache=None, slots_mapping=None, **kwargs):
        batch_size = input_ids.shape[0] if input_ids.dim() == 2 else 1
        cur_pos = position_ids[-1] + 1
        logger.debug(f"batch_size = {batch_size}, cur_pos = {cur_pos}")
        seq_len = self.torch.ones([batch_size], dtype=self.torch.int).to(position_ids.device) * cur_pos
        if slots_mapping is None:
            slots_mapping = self.torch.zeros([batch_size * cur_pos], dtype=self.torch.int)
        if k_cache is None:
            k_cache = self.torch.zeros(self.cache_shape).half()
        if v_cache is None:
            v_cache = self.torch.zeros(self.cache_shape).half()

        model_inputs = {
            FIXED_INPUTS.input_ids: input_ids.npu(),
            FIXED_INPUTS.position_ids: position_ids.npu(),
            FIXED_INPUTS.seq_len: seq_len.npu(),
            FIXED_INPUTS.k_cache: k_cache.npu(),
            FIXED_INPUTS.v_cache: v_cache.npu(),
            FIXED_INPUTS.slots_mapping: slots_mapping.npu(),
        }
        model_inputs.update({kk: vv.npu() for kk, vv in kwargs.items()})

        if self.inv_freq_weight is not None and FIXED_INPUTS.cos_table not in model_inputs:
            logger.debug(f"self.inv_freq_weight.shape = {self.inv_freq_weight.shape}")
            logger.debug(f"position_ids.shape = {position_ids.shape}")
            left = self.inv_freq_weight[None, :, None]
            right = position_ids[:, None, :] if position_ids.dim() == 2 else position_ids[None, None, :]
            freq = (left.to(right.device).float() @ right.float()).transpose(1, 2)
            freq = self.torch.cat([freq, freq], dim=-1)[0].npu()  # has to be float npu values, and dim == 2
            logger.debug(f"freq.shape = {freq.shape}")

            model_inputs[FIXED_INPUTS.cos_table], model_inputs[FIXED_INPUTS.sin_table] = freq.cos(), freq.sin()
        if FIXED_INPUTS.cos_table not in model_inputs:
            raise ValueError("Missing cos_table and sin_table")

        model_inputs.update(self.weights)
        missing_inputs = self.inputs - set(model_inputs.keys())
        if len(missing_inputs) != 0:
            raise ValueError(
                f"Missing inputs: {missing_inputs}, provided: {model_inputs.keys()}, required: {self.inputs}"
            )

        model_outputs = {
            ii: self.torch.ones([batch_size * cur_pos, self.vocab_size]).half().npu() for ii in self.outputs
        }
        bind_map = {FIXED_INPUTS.seq_len: seq_len.cpu()}
        return self.atb_model.forward(model_inputs, model_outputs, bind_map)

    def __call__(self, input_ids, position_ids, k_cache=None, v_cache=None, slots_mapping=None, **kwargs):
        return self.forward(input_ids, position_ids, k_cache, v_cache, slots_mapping, **kwargs)


class ATBModelFromTorch(ATBModel):
    """Create ATB model from raw transformers torch model
    >>> import torch
    >>> from transformers.models.llama import LlamaConfig, LlamaForCausalLM
    >>> from msit_llm.transform.torch_to_atb_python import ATBModel, ATBModelConfig, ATBModelFromTorch
    >>>
    >>> cc = LlamaConfig()
    >>> cc.num_hidden_layers, cc.hidden_size, cc.intermediate_size = 2, 1024, 4096  # smaller
    >>> mm = LlamaForCausalLM(cc)
    >>>
    >>> atb_model = ATBModelFromTorch(mm)
    >>> atb_model.set_weights(mm.state_dict())
    >>> # Also set buffers in, which includes inv_freq. Ignore WARNINGs of missing weights
    >>> atb_model.set_weights(dict(mm.named_buffers()))
    >>> out = atb_model(input_ids=torch.arange(32), position_ids=torch.arange(32))
    >>> print({kk: vv.shape for kk, vv in out.items()})
    # {'output': torch.Size([32, 32000])}
    >>> atb_model.to_file()  # Save atb model to py file
    # 'llamaforcausallm_atb_float.py'
    import llamaforcausallm_atb_float
    bb = llamaforcausallm_atb_float.atb_model()
    atb_model_config=dict(vocab_size=cc.vocab_size, num_attention_heads=cc.num_attention_heads, head_dim=cc.hidden_size // cc.num_attention_heads)
    dd = ATBModel(bb, )

    """
    def __init__(
        self,
        torch_model,
        config=None,
        input_names=BASIC_INPUT_NAMES,
        max_batch_size=1,
        max_seq_len=1024,
        to_quant=False,
        quant_disable_names=None,
    ):
        self.torch_model, self.config = torch_model, config
        self.to_quant, self.quant_disable_names = to_quant, quant_disable_names or ()
        self.config = getattr(torch_model, "config", None) if config is None else config
        if self.config is None:
            raise ValueError("Either config or torch_model.config shold be not empty one")

        self.num_attention_heads = get_config_attr(self.config, "num_attention_heads")
        if self.num_attention_heads is None or self.num_attention_heads <= 0:
            raise ValueError("Invalid config, num_attention_heads not exists or < 0")

        self.hidden_size = get_config_attr(self.config, "hidden_size")
        if self.hidden_size is None or self.hidden_size <= 0:
            raise ValueError("Invalid config, hidden_size not exists or < 0")

        self.vocab_size = get_config_attr(self.config, "vocab_size", -1)
        if self.vocab_size is None or self.vocab_size <= 0:
            raise ValueError("Invalid config, vocab_size not exists or < 0")

        self.head_dim = self.hidden_size // self.num_attention_heads
        self.traced_module = to_transformers_traced_module(torch_model, input_names=input_names)
        self.model_name = torch_model.__class__.__name__

        self.weight_names, self.weight_stack_map = list(self.traced_module.state_dict().keys()), {}
        for ii in self.weight_names:
            self.weight_stack_map.setdefault(".".join(ii.split(".")[:-1]), []).append(ii)

        self.torch_module_to_atb_map = {}
        for kk, vv in TORCH_MODULE_TO_ATB_MAP.items():
            re_key = re.compile(kk)
            if vv["op_type"] == "SelfAttention":
                vv["op_param"].update({"headNum": self.num_attention_heads, "kvHeadNum": self.num_attention_heads})
            self.torch_module_to_atb_map[re_key] = Operation(**vv)

        self.pre_query_name, self.pre_key_name, self.pre_value_name, self.is_apply_rope = "", "", "", False
        self.model_inputs, self.model_outputs, self.operations = [], [], []

        self.convert_fx_traced_module()
        if to_quant:
            logger.info(f"calling convert_to_quant, quant_disable_names = {self.quant_disable_names}")
            self.convert_to_quant()
        self.to_file()
        self.atb_model = self.build_atb_model()

        self.atb_model_config = ATBModelConfig(
            self.vocab_size, self.num_attention_heads, self.head_dim, max_batch_size, max_seq_len
        )
        super().__init__(atb_model=self.atb_model, atb_model_config=self.atb_model_config)

    @staticmethod
    def _get_module_type_by_nn_module_stack(node):
        node_module_stack = list(node.meta[NN_MODULE_STACK].values())
        return None if len(node_module_stack) == 0 else node_module_stack[-1].__name__

    @staticmethod
    def _get_module_name_by_nn_module_stack(node):
        node_module_stack = list(node.meta[NN_MODULE_STACK].keys())
        return None if len(node_module_stack) == 0 else node_module_stack[-1]

    @staticmethod
    def _should_skip_node(node):
        if node.op == FX_OP_TYPES.call_method and node.target in SKIP_NODES:
            return True
        if node.op == FX_OP_TYPES.call_function and node.target.__name__ in SKIP_NODES:
            return True
        if node.op == FX_OP_TYPES.call_function and node.meta.get("is_wrapped", False):
            return True
        return False

    def _find_in_torch_module_to_atb_map(self, node_type):
        for kk, vv in self.torch_module_to_atb_map.items():
            if kk.fullmatch(node_type):
                return vv.copy()
        return None

    def _convert_module(self, node_module_type, node_module_name, input_names):
        atb_operation = self._find_in_torch_module_to_atb_map(node_module_type)
        outputs = [node_module_name + ".out"]
        atb_operation.op_name = node_module_name
        atb_operation.inputs = getattr(atb_operation, "inputs", []) + input_names
        atb_operation.outputs = getattr(atb_operation, "outputs", []) + outputs
        return atb_operation

    def _get_node_type_and_inputs_and_name(self, node, output_node_map={}):
        cur_module_name = self._get_module_name_by_nn_module_stack(node)
        if node.op == FX_OP_TYPES.call_function and self._find_in_torch_module_to_atb_map(node.target.__name__):
            node_module_type = node.target.__name__
            # No other inputs if function
            cur_inputs = []
            for ii in node.all_input_nodes:
                if ii.name not in output_node_map:
                    continue
                cur_inputs += output_node_map[ii.name]
            module_name = cur_module_name + "." + node.name
        else:
            node_module_type = self._get_module_type_by_nn_module_stack(node)
            # None marks for placeholder of other inputs
            cur_inputs = self.weight_stack_map.get(cur_module_name, []) + [None]
            module_name = cur_module_name
        return node_module_type, cur_inputs, module_name

    def _check_and_set_pre_qkv_name(self, atb_operation):
        if atb_operation.op_type == "Linear":
            sub_name = atb_operation.op_name.split(".")[-1]
            if "q" in sub_name:
                self.pre_query_name = atb_operation.outputs[0]
            elif "k" in sub_name:
                self.pre_key_name = atb_operation.outputs[0]
            elif "v" in sub_name:
                self.pre_value_name = atb_operation.outputs[0]

    def _op_process_linear(self, atb_operation=None, module_name=""):
        bias_name = f"{atb_operation.op_name}.bias"
        if bias_name in self.weight_names:
            atb_operation.inputs.insert(1, bias_name)
            atb_operation.op_param.update({"hasBias": True})
        self.operations.append(atb_operation)

    def _op_process_rope(self, atb_operation=None, module_name=""):
        self.is_apply_rope = True
        self.model_inputs += [FIXED_INPUTS.cos_table, FIXED_INPUTS.sin_table]
        self.operations.append(
            self._convert_module("Gather", "gather_cos", [FIXED_INPUTS.cos_table, FIXED_INPUTS.position_ids])
        )
        self.operations.append(
            self._convert_module("Gather", "gather_sin", [FIXED_INPUTS.sin_table, FIXED_INPUTS.position_ids])
        )

        if FIXED_INPUTS.position_ids not in self.model_inputs:
            self.model_inputs += [FIXED_INPUTS.position_ids]

    def _op_process_attention(self, atb_operation=None, module_name=""):
        atb_operation.inputs = [
            module_name + ".q_embed_",
            module_name + ".k_embed_",
            module_name + ".v_embed_",
            FIXED_INPUTS.seq_len,
        ]

        if self.is_apply_rope:
            inputs = [self.pre_query_name, self.pre_key_name, "gather_cos.out", "gather_sin.out", FIXED_INPUTS.seq_len]
            outputs = [module_name + ".q_embed", module_name + ".k_embed"]
            self.operations.append(
                Operation(
                    op_type="Rope",
                    op_param={"rotaryCoeff": 2},
                    inputs=inputs,
                    outputs=outputs,
                    op_name=module_name + ".rope",
                )
            )

        reshape_and_cache_inputs = [
            module_name + ".k_embed_",
            module_name + ".v_embed_",
            FIXED_INPUTS.k_cache,
            FIXED_INPUTS.v_cache,
            FIXED_INPUTS.slots_mapping,
        ]
        num_attention_heads, head_dim = self.num_attention_heads, self.head_dim  # Essential for `to_file`
        self.operations += [
            Operation(
                op_type="add_reshape",
                function=lambda org_shape: [org_shape[0], num_attention_heads, head_dim],
                inputs=[module_name + ".q_embed"],
                outputs=[module_name + ".q_embed_"],
                op_name=module_name + ".q." + RESHPAE_KIND.reshape_qkv,
            ),
            Operation(
                op_type="add_reshape",
                function=lambda org_shape: [org_shape[0], num_attention_heads, head_dim],
                inputs=[module_name + ".k_embed"],
                outputs=[module_name + ".k_embed_"],
                op_name=module_name + ".k." + RESHPAE_KIND.reshape_qkv,
            ),
            Operation(
                op_type="add_reshape",
                function=lambda org_shape: [org_shape[0], num_attention_heads, head_dim],
                inputs=[self.pre_value_name],
                outputs=[module_name + ".v_embed_"],
                op_name=module_name + ".v." + RESHPAE_KIND.reshape_qkv,
            ),
            Operation(
                op_type="ReshapeAndCache",
                op_param={},
                inputs=reshape_and_cache_inputs,
                outputs=[FIXED_INPUTS.k_cache, FIXED_INPUTS.v_cache],
                op_name=module_name + ".reshape_and_cache",
            ),
            atb_operation,
        ]

        if FIXED_INPUTS.k_cache not in self.model_inputs:
            self.model_inputs += [
                FIXED_INPUTS.k_cache,
                FIXED_INPUTS.v_cache,
                FIXED_INPUTS.slots_mapping,
                FIXED_INPUTS.seq_len,
            ]

    def _refine_inputs_outputs(self, input_node_map, output_node_map, operation_outputs):
        gathered_module_inputs = {}
        for kk, vv in input_node_map.items():
            gathered_inputs = []
            for ii in vv:
                if ii in output_node_map:
                    gathered_inputs += output_node_map[ii]
            for ii in set(gathered_inputs):
                if ii == kk or ii == kk + ".out":
                    continue
                gathered_module_inputs.setdefault(kk, []).extend(
                    operation_outputs.get(ii, ii if isinstance(ii, list) else [ii])
                )

        for op in self.operations:
            if not isinstance(op, list) and (len(op.inputs) == 0 or op.inputs[-1] is not None):
                continue

            op.inputs = op.inputs[:-1]  # Exclude last None
            if op.is_weights_first:
                op.inputs = op.inputs + gathered_module_inputs.get(op.op_name, [])
            else:
                op.inputs = gathered_module_inputs.get(op.op_name, []) + op.inputs

    def convert_fx_traced_module(self):
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

            cur_module_name = self._get_module_name_by_nn_module_stack(node)
            input_node_map.setdefault(cur_module_name, []).extend([ii.name for ii in node.all_input_nodes])

            logger.debug(f"cur_module_name = {cur_module_name}, previous_module_name = {previous_module_name}")
            if cur_module_name != base_module_name:
                output_node_map[node.name] = previous_operation_out  # will be overwriten later
            if cur_module_name == previous_module_name:
                continue
            if self._should_skip_node(node):
                continue
            previous_module_name = cur_module_name

            logger.debug(f"cur_module_name = {cur_module_name}, node.name = {node.name}")
            node_module_type, cur_inputs, module_name = self._get_node_type_and_inputs_and_name(node, output_node_map)
            if self._find_in_torch_module_to_atb_map(node_module_type) is None:
                logger.warning(f"node not supported: node.name = {node.name}, node.type = {node_module_type}")
                continue
            if len(cur_inputs) == 0:
                logger.warning(f"found none valid input: node.name = {node.name}, node.type = {node_module_type}")
                continue

            atb_operation = self._convert_module(node_module_type, module_name, cur_inputs)
            self._check_and_set_pre_qkv_name(atb_operation)
            logger.debug(f"atb_operation = {atb_operation.to_json()}")
            if not self.is_apply_rope and atb_operation.op_type == "Rope":
                self._op_process_rope(atb_operation=atb_operation, module_name=module_name)
            elif atb_operation.op_type == "SelfAttention":
                self._op_process_attention(atb_operation=atb_operation, module_name=module_name)
                cur_outputs = self.operations[-1].outputs
                output_node_map[node.name] = previous_operation_out = operation_outputs[cur_module_name] = cur_outputs
            elif atb_operation.op_type in ["Linear", "LinearParallel"]:
                self._op_process_linear(atb_operation=atb_operation, module_name=module_name)
                cur_outputs = self.operations[-1].outputs
                output_node_map[node.name] = previous_operation_out = operation_outputs[cur_module_name] = cur_outputs
            else:
                self.operations.append(atb_operation)
                cur_outputs = atb_operation.outputs
                output_node_map[node.name] = previous_operation_out = operation_outputs[cur_module_name] = cur_outputs

        self._refine_inputs_outputs(input_node_map, output_node_map, operation_outputs)
        self.operations[-1].outputs = self.model_outputs
        return self.model_inputs, self.model_outputs, self.weight_names, self.operations

    def convert_to_quant(self):
        # Has to split out from convert_fx_traced_module, needs actual Linear input names
        operations_with_quant = []
        for op_id, op in enumerate(self.operations):
            logger.debug(f"op.op_name = {op.op_name}")
            if op.op_type not in ["Linear", "LinearParallel"]:
                operations_with_quant.append(op)
                continue
            if op.op_name in self.quant_disable_names:
                operations_with_quant.append(op)
                continue

            linear_quant_weights = []
            bias, deq_scale = f"{op.op_name}.bias", f"{op.op_name}.deq_scale"
            input_scale, input_offset = f"{op.op_name}.input_scale", f"{op.op_name}.input_offset"
            self.weight_names += [input_scale, input_offset, deq_scale]

            if bias not in op.inputs:
                linear_quant_weights.append(bias)
                self.weight_names.append(bias)
            linear_quant_weights.append(deq_scale)

            elewise_quant_node = Operation(
                op_type="Elewise",
                op_param={"elewiseType": "ELEWISE_QUANT_PER_CHANNEL"},
                op_name=f"{op.op_name}.elewise_quant",
                inputs=[f"{op.inputs[0]}", input_scale, input_offset],
                outputs=[f"{op.op_name}.elewise_quant.out"],
            )

            op.op_param.update({"transposeA": False, "transposeB": True, "hasBias": True, "outDataType": "ACL_FLOAT16"})
            op.inputs = elewise_quant_node.outputs + op.inputs[1:] + linear_quant_weights

            operations_with_quant += [elewise_quant_node, op]
        self.operations = operations_with_quant

    def build_atb_model(self):
        import torch
        import torch_npu

        atb_speed_path = os.getenv("ATB_SPEED_HOME_PATH", None)
        if not atb_speed_path:
            raise OSError("ATB_SPEED_HOME_PATH environment variable not valid, try install mindie")
        sys.path.append(os.path.join(atb_speed_path, "lib"))
        try:
            import _libatb_torch as atb  # May error
        except (ImportError, ModuleNotFoundError) as err:
            raise OSError("import _libatb_torch failed, try install mindie") from err

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
                    operation=atb._BaseOperation(
                        op_type=op.op_type, op_param=json.dumps(op.op_param), op_name=op.op_name
                    ),
                    input=op.inputs,
                    output=op.outputs,
                )
        atb_model.build()
        atb_model.execute_as_single = False
        return atb_model

    def to_file(self, output_file=None):
        indent = "    "
        in_tensors = self.model_inputs + self.weight_names
        operation_inputs = [jj for op in self.operations for jj in op.inputs]
        valid_inputs = [ii for ii in in_tensors if ii in operation_inputs]
        intensors_str = "[\n" + "".join([f"{indent}{indent}'{ii}',\n" for ii in valid_inputs]) + f"{indent}]"

        contents = "\n".join(
            [
                "import os",
                "import sys",
                "import json",
                "import torch",
                "import torch_npu",
                "",
                "path = os.getenv('ATB_SPEED_HOME_PATH')",
                "sys.path.append(os.path.join(path, 'lib'))",
                "import _libatb_torch as atb",
                "",
                "def atb_model():" "",
                f"{indent}num_attention_heads, head_dim = {self.num_attention_heads}, {self.head_dim}",
                f"{indent}atb_model = atb._GraphOperation('{self.model_name}')",
                f"{indent}in_tensors = {intensors_str}",
                f"{indent}atb_model.add_input_output(input=in_tensors, output={self.operations[-1].outputs})",
                "",
                "",
            ]
        )

        for op in self.operations:
            if op.op_type == "add_reshape":
                function = get_lambda_source_code(op.function)
                contents += f"{indent}atb_model.add_reshape('{op.inputs[0]}', '{op.outputs[0]}', {function})\n"
            else:
                op_kwargs = f"op_type='{op.op_type}', op_param='{json.dumps(op.op_param)}', op_name='{op.op_name}'"
                contents += f"{indent}atb_model.add_operation(\n{indent}{indent}"
                contents += f",\n{indent}{indent}".join(
                    [
                        f"operation=atb._BaseOperation({op_kwargs})",
                        f"input={op.inputs}",
                        f"output={op.outputs}",
                    ]
                )
                contents += f",\n{indent})\n"
        contents += f"{indent}atb_model.build()\n"
        contents += f"{indent}atb_model.execute_as_single = False\n"
        contents += f"{indent}return atb_model"

        if output_file is None:
            output_file = "_".join([self.model_name.lower(), "atb", "quant.py" if self.to_quant else "float.py"])
        elif not output_file.endswith(".py"):
            output_file += ".py"
        with open(output_file, "w") as ff:
            ff.write(contents)
        return output_file


def transform(
    source_path,
    input_names=BASIC_INPUT_NAMES,
    output_file=None,
    to_quant=False,
    quant_disable_names=None,
):
    logger.info("Building model using transformers...")
    model, config = build_model(source_path)

    logger.info("Transforming to atb")
    atb_model = ATBModelFromTorch(
        torch_model=model,
        config=config,
        input_names=input_names,
        to_quant=to_quant,
        quant_disable_names=quant_disable_names,
    )
    output_file = atb_model.to_file(output_file=output_file)

    logger.info("=" * 30)
    logger.info(f"Saved to: {output_file}\n")

    logger.info("=" * 30)
    logger.info(f"atb_model config:\n{atb_model.atb_model_config.to_dict()}\n")

    model_name = os.path.splitext(os.path.basename(output_file))[0]
    logger.info("=" * 30)
    logger.info(
        """Run like:

    python3 -c "
    import torch, torch_npu
    import {model_name}
    from msit_llm.transform.torch_to_atb_python import ATBModel, ATBModelConfig

    atb_model = {model_name}.atb_model()
    weights = torch.load(\'$WEIGHT_PATH\')  # Use actual WEIGHT_PATH

    vocab_size, num_attention_heads, head_dim = {vocab_size}, {num_attention_heads}, {head_dim}
    cc = ATBModelConfig(vocab_size=vocab_size, num_attention_heads=num_attention_heads, head_dim=head_dim)
    aa = ATBModel(atb_model, cc)
    aa.set_weights(weights)
    
    input_len = 32
    out = aa.forward(
        input_ids=torch.arange(input_len),
        position_ids=torch.arange(input_len),
        cos_table=torch.rand(input_len, head_dim),
        sin_table=torch.rand(input_len, head_dim),
    )
    print(out)
    "
    """.format(
            model_name=model_name,
            vocab_size=atb_model.atb_model_config.vocab_size,
            num_attention_heads=atb_model.atb_model_config.num_attention_heads,
            head_dim=atb_model.atb_model_config.head_dim,
        )
        .replace(" " * 8, "##")  # Replace the inner 8 ' ' to '##' as a mark
        .replace(" " * 4, "")  # Remove all 4 ' '
        .replace("##", " " * 4)  # Replace the marked '##' back to 4 ' '
    )
