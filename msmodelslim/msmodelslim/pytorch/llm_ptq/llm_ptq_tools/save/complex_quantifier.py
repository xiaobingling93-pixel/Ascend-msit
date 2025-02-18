# Copyright Huawei Technologies Co., Ltd. 2025. All rights reserved.
import torch
import numpy as np

# KIA part
from msmodelslim.pytorch.llm_sparsequant.atomic_power_outlier import quant_one_weight_by_outliers
from msmodelslim.pytorch.lowbit.calibration import LlamaRMSNormBias
from msmodelslim.pytorch.lowbit.quant_modules import LinearQuantizer as LowBitLinearQuantizer
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_funcs import fake_quantize_save

from msmodelslim.pytorch.llm_sparsequant.sparsequant_modules import LinearSparseQuantizer
from msmodelslim.pytorch.llm_ptq.anti_outlier.graph_utils import NormBias
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_modules import LinearQuantizer, LinearNf4Quantizer
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.llm_ptq_utils import QuantType
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.fa_quant import (
    export_fa_quant_params,
    is_attn_module_and_then_check_quantizer
)
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.simulate_tp import ParallelLinearCol


def deqscale_process(input_scale, scale):
    deq_scale = input_scale * scale
    if deq_scale.ndim > 1:
        deq_scale = deq_scale.squeeze(1)
    deq_scale = deq_scale.cpu()
    return deq_scale


def change_bias(fp_weight, module):
    if module.bias is None:
        bias_shape = fp_weight.shape[0]
        fp_bias = torch.zeros(bias_shape)
    else:
        fp_bias = module.bias.cpu()
    return fp_bias


def deqscale2int64(scale):
    scale = scale.numpy()
    scale = np.frombuffer(scale.tobytes(), dtype=np.int32).astype(np.int64)
    scale = torch.tensor(scale)
    return scale


def deqscale2int64_by_dtype(scale, is_bf16):
    if is_bf16:
        return scale
    else:
        return deqscale2int64(scale)


def _get_module_quant_input(module):
    fp_weight = module.weight.cpu()
    weight_scale, weight_offset = module.quant_weight.weight_scale, module.quant_weight.weight_offset
    device = weight_scale.device if weight_scale is not None else None
    scale = weight_scale.cpu() if weight_scale is not None else None
    offset = weight_offset.cpu() if weight_offset is not None else None
    round_opt = False if isinstance(module, LowBitLinearQuantizer) else module.quant_weight.round_opt
    ret = fp_weight, device, scale, offset, round_opt
    return ret


class ComplexQuantifier:
    def __init__(self, cfg, is_deepseek_v2, rollback_names, torch_dtype, is_inner_norm_used):
        self.cfg = cfg
        self.is_deepseek_v2 = is_deepseek_v2
        self.rollback_names = rollback_names
        self.torch_dtype = torch_dtype
        self.is_inner_norm_used = is_inner_norm_used

        self.skip_module_names = set()

    def generate_weight_of_module(self, name: str, module: torch.nn.Module):
        if name in self.skip_module_names:
            return

        model_quant_type = self.cfg.model_quant_type
        if self.cfg.use_fa_quant and is_attn_module_and_then_check_quantizer(module, name):
            yield from self.generate_weight_of_fa_module(name, module)
        elif isinstance(module, LinearNf4Quantizer):
            yield from self.generate_weight_of_nf_module(name, module)
        elif isinstance(module, ParallelLinearCol):
            yield from self.generate_weight_of_tp_module(name, module, model_quant_type)
        # 处理 Norm 对应的 weight、bias
        elif isinstance(module, (NormBias, LlamaRMSNormBias)):
            yield from self.generate_weight_of_norm_module(name, module, model_quant_type)
        # 处理Linear、以及附属scale、offset等params
        elif isinstance(module, (LinearQuantizer, LinearSparseQuantizer, LowBitLinearQuantizer)):
            yield from self.generate_weight_of_linear_module(name, module, model_quant_type)
        else:
            for key, param in module.named_parameters(name, recurse=False):
                yield key, QuantType.FLOAT, param

    def generate_weight_of_nf_module(self, name, module):
        yield name + '.weight', QuantType.NF4, module.weight
        if module.bias is not None:
            yield name + '.bias', QuantType.NF4, module.bias

    def generate_weight_of_fa_module(self, name, module):
        quant_param_scale, quant_param_offset, _ = export_fa_quant_params(module, name)
        for name, param in quant_param_scale.items():
            yield name, QuantType.FAQuant, param
        for name, param in quant_param_offset.items():
            yield name, QuantType.FAQuant, param

    def generate_weight_of_tp_module(self, name, module, model_quant_type):
        quant_param, _ = module.get_quant_param()
        if hasattr(self.cfg, 'tp_size'):
            self.concat_simulate_linear(name, module, quant_param)
        for name, param in quant_param:
            yield name, model_quant_type, param

    def generate_weight_of_norm_module(self, name, module, model_quant_type):
        # 不暴露原norm
        self.skip_module_names.add(name + '.module')

        is_norm_bias = isinstance(module, NormBias)
        anti_norm_weight: torch.Tensor = module.module.weight.cpu() if is_norm_bias else module.weight.cpu()
        anti_norm_bias: torch.Tensor = module.bias.cpu()
        anti_norm_name_weight = name + '.module.weight' if self.is_inner_norm_used else name + '.weight'
        anti_norm_name_bias = name + '.module.bias' if self.is_inner_norm_used else name + '.bias'
        yield anti_norm_name_weight, model_quant_type, anti_norm_weight.clone().detach()
        yield anti_norm_name_bias, model_quant_type, anti_norm_bias.clone().detach()
        if self.is_inner_norm_used:
            yield name + '.weight', QuantType.FLOAT, module.weight.cpu()

    def generate_weight_of_linear_module(self, name, module, model_quant_type):
        if not module.quant_weight.is_enable:
            return

        quant_weight, fp_weight, weight_scale, weight_offset = self.get_param_from_quantizer(module)
        if quant_weight is None:
            return

        # 所有专家层都使用动态量化
        if "mlp" in name and self.is_deepseek_v2 and model_quant_type is QuantType.W8A8:
            model_quant_type = QuantType.W8A8_DYNAMIC

        # 各种量化均需要提供 weight
        quant_weight: torch.Tensor = quant_weight.to(device=fp_weight.device)
        save_quant_weight = quant_weight.cpu().to(torch.int8)
        yield name + '.weight', model_quant_type, save_quant_weight

        # W4A16/W8A16 需要提供 weight_scale、weight_offset
        if model_quant_type in [QuantType.W8A16, QuantType.W4A16, QuantType.W8A8_DYNAMIC, QuantType.W8A8]:
            yield name + '.weight_scale', model_quant_type, weight_scale.cpu()
            yield name + '.weight_offset', model_quant_type, weight_offset.cpu()

        # W8A8/W8A8S 需要提供 deq_scale、quant_bias、input_scale、input_offset
        if model_quant_type in [QuantType.W8A8, QuantType.W8A8S]:
            input_scale = module.quant_input.input_scale
            input_offset = module.quant_input.input_offset
            yield name + '.input_scale', model_quant_type, input_scale.cpu()
            yield name + '.input_offset', model_quant_type, input_offset.cpu()

            input_scale = input_scale.to(device=quant_weight.device)
            input_offset = input_offset.to(device=quant_weight.device)
            deq_scale = deqscale_process(input_scale, weight_scale).to(torch.float32)
            quant_weight = quant_weight.to(torch.float32)
            input_offset = input_offset.to(torch.float32)
            correction = (quant_weight.sum(dim=1) * input_offset).cpu()
            fp_bias = change_bias(fp_weight, module)
            quant_bias = torch.round(fp_bias / deq_scale - correction)
            deq_scale = deqscale2int64_by_dtype(deq_scale,
                                                self.torch_dtype == torch.bfloat16)

            yield name + '.quant_bias', model_quant_type, quant_bias.cpu().to(torch.int32)
            yield name + '.deq_scale', model_quant_type, deq_scale.cpu()

    def concat_simulate_linear(self, name, module, quant_param):
        if name in self.rollback_names:
            return
        if module.cfg.model_quant_type == QuantType.FLOAT:
            return
        concat_weight_list = []
        for tp_index in range(module.cfg.tp_size):
            concat_name = '.'.join([name, f'tp_list', str(tp_index), 'weight'])
            concat_weight_list.append(quant_param.get(concat_name))
        concat_weight = torch.cat(concat_weight_list, dim=-1)
        quant_param[name + '.weight'] = concat_weight

    @torch.no_grad()
    def get_param_from_quantizer(self, module):
        quant_weight = None
        fp_weight, device, weight_scale, weight_offset, round_opt = _get_module_quant_input(module)
        if isinstance(module, LinearQuantizer):
            quant_weight, _ = fake_quantize_save(fp_weight, weight_scale, weight_offset, bit=8,
                                                 round_opt=round_opt, device=device)
        if isinstance(module, LinearSparseQuantizer):
            _, _, quant_weight, _ = quant_one_weight_by_outliers(
                fp_weight, powerquant=self.cfg.nonuniform, fraction=self.cfg.fraction, num_bits=self.cfg.w_bit,
                per_channel=not self.cfg.mm_tensor)
        if isinstance(module, LowBitLinearQuantizer):
            fp_weight = module.fp_weight
            if module.disable_input:
                res = None, fp_weight, weight_scale, weight_offset
                return res
            if self.cfg.model_quant_type == QuantType.W8A8S:
                bit = 8
            else:
                bit = self.cfg.w_bit
            quant_weight, _ = fake_quantize_save(fp_weight, weight_scale, weight_offset, bit=bit,
                                                 round_opt=round_opt, device=module.weight.device,
                                                 group_size=self.cfg.group_size)
        res = quant_weight, fp_weight, weight_scale, weight_offset
        return res
