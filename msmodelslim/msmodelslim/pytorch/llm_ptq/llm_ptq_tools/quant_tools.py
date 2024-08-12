# Copyright Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

from __future__ import absolute_import, division, print_function

import re
import os
import gc
import copy
import functools
from collections import defaultdict

from tqdm import tqdm
import torch
import numpy as np
import torch.nn as nn
from safetensors.torch import save_file

from accelerate.hooks import add_hook_to_module, remove_hook_from_module

from msmodelslim import logger as msmodelslim_logger
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_config import QuantConfig
from ascend_utils.common.security import (get_valid_write_path, SafeWriteUmask, check_element_type,
    check_type, check_dict_element, get_write_directory)
from msmodelslim.pytorch.llm_ptq.anti_outlier.graph_utils import NormBias, extract_dag, input_to_cpu, norm_class_detect
# KIA part
from msmodelslim.pytorch.llm_ptq.anti_outlier.anti_outlier import deepcopy_model
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_modules import (
    Quantizer, Conv2dQuantizer, LinearQuantizer, layer_wise_calib
)
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_funcs import (
    fake_quantize_save, get_features, linear_quantization_params
)
from msmodelslim.pytorch.llm_sparsequant.atomic_power_outlier import quant_one_weight_by_outliers
from msmodelslim.pytorch.llm_sparsequant.sparsequant_modules import LinearSparseQuantizer
from msmodelslim.pytorch.lowbit.atomic_power_outlier import \
    quant_one_weight_by_outliers as quant_one_weight_by_outliers_low_bit
from msmodelslim.pytorch.lowbit.calibration import (
    replace_RMSNorm, QuantXDecoderLayer, LlamaRMSNormBias
)
from msmodelslim.pytorch.lowbit.quant_modules import LinearQuantizer as LowBitLinearQuantizer
from msmodelslim.pytorch.lowbit.quant_modules import Quantizer as LowBitQuantizer
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.llm_ptq_utils import QuantType, WeightQuantMethod
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.llm_ptq_utils import QuantModelJsonDescription
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.llm_ptq_utils import (
    SAVE_TYPE_LIST,
    SAVE_TYPE_NUMPY,
    SAVE_TYPE_SAFE_TENSOR,
)

from msmodelslim.pytorch.llm_ptq.anti_outlier.dag_utils.torch_dag_adapter import TorchDAGAdapter

HF_HOOK = "_hf_hook"


class Calibrator(object):
    """ Calibrator for post-training quantization."""

    def __init__(self, model,
                 cfg: QuantConfig,
                 calib_data=None,
                 disable_level='L0',
                 all_tensors=None):
        # calib_data的类型校验方法在get_calib_data方法中
        # The type checking of calib_data is in the method of get_calib_data
        check_type(model, nn.Module, param_name="model")
        check_type(cfg, QuantConfig, param_name="cfg")
        check_type(disable_level, str, param_name='disable_level')
        if all_tensors:
            check_element_type(all_tensors, torch.Tensor, dict, param_name="all_tensors")

        self.cfg = cfg
        self.logger = msmodelslim_logger
        self.calib_data = self.get_calib_data([]) if calib_data is None else self.get_calib_data(calib_data)
        self.use_kvcache_quant = cfg.use_kvcache_quant
        self.norm_class_name = cfg.norm_class_name 

        self.quant_param_dict = {}
        # 记录被量化module名称，相关的scale、offset等参数名称 key:weight的名称， value:scale、offset等参数的名称
        self.quantized_module_param_dict = defaultdict(list)

        if self.use_kvcache_quant:
            self.get_kvcache(model)

        # 记录浮点模型权重
        if hasattr(model, 'ori_state_dict'):
            self.ori_fp_weight = getattr(model, 'ori_state_dict')
        else:
            self.ori_fp_weight = model.state_dict()
        if self.cfg.do_smooth:
            replace_RMSNorm(model)
        check_dict_element(self.ori_fp_weight, value_type=torch.Tensor, param_name='ori_fp_weight',
                           additional_msg="Get origin model weight failed, please check your model.")
        # 初始化模型权重json描述
        self.quant_model_json_description = QuantModelJsonDescription(self.cfg.model_quant_type, 
                                                                      self.cfg.use_kvcache_quant)
        if not re.match(r'^L((?!0)\d+|0)$', disable_level):
            raise ValueError('Please check your disable_level config.')
        self.disable_level = disable_level

        model = self.init_model_device(model)
        self.last_layer_name = None
        self.rollback_names = None
        self.act_states = None
        self.rollback_names_process(model)
        if self.cfg.calib_mode == 1:
            if (self.cfg.a_bit <= 8) or self.cfg.w_hessian:
                self.all_tensors = all_tensors
            else:
                self.all_tensors = None
        try:
            self.model_with_accelerate = judge_model_with_accelerate(model)
        except Exception as e:
            raise Exception("Please check your model and config.", e) from e
        if not self.model_with_accelerate:
            self.device_org = next(model.parameters()).device
        try:
            self.model = self.quantize(model)
            if self.calib_data:
                self.enable_quant()
        except Exception as e:
            raise Exception("Please check your model and config.", e) from e
        self.logger.info("Wrap Quantizer success!")

    def init_model_device(self, model):
        if self.cfg.device == "cpu":
            same_device = self.cfg.device == model.device.type
        else:
            same_device = self.cfg.device == model.device
        if not same_device:
            self.logger.warning("Model is not on the deivce indicated in `QuantConfig`, "
                                "Model is on the device `{}` while `QuantConfig` "
                                "indicates `{}`".format(model.device, self.cfg.device))
            self.logger.info("Transfering model from `{}` to `{}`...".format(model.device, self.cfg.device))
            model = model.to(self.cfg.device)
            self.logger.info("Transfer done. Suggest to check model and calib_data (if provided) on the "
                             "device that `QuantConfig` indicates.")
        return model

    def get_kvcache(self, model):
        kv_linears, num_kv = self.dag_extract_kvlinears(model)
        kv_cache = self.get_kvcache_features(model, kv_linears, num_kv)
        self.get_kvcache_quant_param(kv_cache, num_kv)

    def dag_extract_kvlinears(self, model):
        model_cpu = deepcopy_model(model).float()
        dummy_input = input_to_cpu(self.calib_data[0])

        norm_class = norm_class_detect(model_cpu)
        dag = extract_dag(model_cpu, dummy_input, hook_nodes=norm_class)
        kv_linears, num_kv = dag.get_kv_linears()

        del model_cpu
        return kv_linears, num_kv
    
    def get_kvcache_features(self, model, kv_linears, num_kv):
        model.eval()
        kv_cache = {}

        def update_key_value_extremums(kv_cache, name, max_min, in_hidden_size, chunk_size):
            comming_max, comming_min = max_min[0], max_min[1]
            # 分块操作
            if chunk_size == 2:
                key_max, value_max = torch.chunk(comming_max, chunk_size, dim=0)
                key_min, value_min = torch.chunk(comming_min, chunk_size, dim=0)
            elif chunk_size == 3:
                res_dim = comming_max.shape[-1] - in_hidden_size

                _, kv_max = torch.split(comming_max, [in_hidden_size, res_dim])
                _, kv_min = torch.split(comming_min, [in_hidden_size, res_dim])
                key_max, value_max = torch.chunk(kv_max, 2, dim=0)
                key_min, value_min = torch.chunk(kv_min, 2, dim=0)

            # 获取k_name和v_name
            k_name = 'k_proj'
            v_name = 'v_proj'

            # 更新key极值
            update_extremum(kv_cache[name], k_name, 'max', torch.max, key_max)
            update_extremum(kv_cache[name], k_name, 'min', torch.min, key_min)
            # 更新value极值
            update_extremum(kv_cache[name], v_name, 'max', torch.max, value_max)
            update_extremum(kv_cache[name], v_name, 'min', torch.min, value_min)

        def update_extremum(kv_cache, name, key, torch_function, value):
            if name not in kv_cache:
                kv_cache[name] = {}
                
            if key in kv_cache[name]:
                kv_cache[name][key] = torch_function(kv_cache[name][key], value)
            else:
                kv_cache[name][key] = value

        def kv_tensor(name, in_hidden_size, tensor):
            if name not in kv_cache:
                kv_cache[name] = {}

            hidden_dim = tensor.shape[-1]
            tensor = tensor.view(-1, hidden_dim).detach()
            comming_max = torch.max(tensor, dim=0)[0]  
            comming_min = torch.min(tensor, dim=0)[0]  
            max_min = [comming_max, comming_min]

            if num_kv == 2:
                # 更新最大值
                update_extremum(kv_cache, name, 'max', torch.max, comming_max)
                # 更新最小值
                update_extremum(kv_cache, name, 'min', torch.min, comming_min)
            elif num_kv == 1:
                update_key_value_extremums(kv_cache, name, max_min, in_hidden_size, 2)
            elif num_kv == 0:
                update_key_value_extremums(kv_cache, name, max_min, in_hidden_size, 3)

        def kv_cache_hook(m, x, y, name):
            if isinstance(x, tuple):
                x = x[0]
            if isinstance(y, tuple):
                y = y[0]
            kv_tensor(name, x.shape[-1], y)
        
        hooks = []
        for name, m in model.named_modules():
            if isinstance(m, nn.Linear):
                if name in kv_linears:
                    hooks.append(
                    m.register_forward_hook(
                        functools.partial(kv_cache_hook, name=name))
                )
        
        for data in tqdm(self.calib_data):
            if isinstance(data, tuple) or isinstance(data, list):
                with torch.no_grad():
                    model(*data)
            elif isinstance(data, dict):
                with torch.no_grad():
                    model(**data)
        for h in hooks:
            h.remove()
        
        return kv_cache

    def get_kvcache_quant_param(self, kv_cache, num_kv):
        if num_kv == 2:
            for key in kv_cache.keys():
                self.get_quant_module_param(key, key, kv_cache[key])
        else:
            for key in kv_cache.keys():
                for kv_key in kv_cache[key].keys():
                    new_key = key + '.' + kv_key
                    self.get_quant_module_param(key, new_key, kv_cache[key][kv_key])

    def get_quant_module_param(self, key, new_key, kv_cache):
        key_weight = key + '.weight'
        new_key_scale, new_key_offset = new_key + '.kv_cache_scale', new_key + '.kv_cache_offset'

        scale, zero_point = linear_quantization_params(8, kv_cache['min'], kv_cache['max'], 
                                                       integral_zero_point=True, q_signed=True, sym=True)

        self.quant_param_dict[new_key_scale] = scale.to('cpu')
        self.quant_param_dict[new_key_offset] = zero_point.to('cpu')
        self.quantized_module_param_dict[key_weight].append(new_key_scale)
        self.quantized_module_param_dict[key_weight].append(new_key_offset)

    def rollback_names_process(self, model):
        # 自动回退lm_head层
        quant_name_list = []
        for name, module in list(model.named_modules()):
            if isinstance(module, (nn.Linear, nn.modules.linear.NonDynamicallyQuantizableLinear)):
                quant_name_list.append(name)
        if quant_name_list:
            last_layer_name = quant_name_list[-1]
        else:
            raise ValueError("No nn.Linear or nn.Conv2D found in model, please check it.")

        # 校验用户指定的回退层是否存在
        for name in self.cfg.disable_names:
            if name not in quant_name_list:
                raise ValueError(f"disable_names {name} is invalid, please check it.")
        if self.cfg.disable_last_linear:
            self.logger.info(f"Automatic disable last layer name: {last_layer_name}")
            self.rollback_names = list(set(self.cfg.disable_names + [last_layer_name]))
        else:
            self.rollback_names = list(set(self.cfg.disable_names))

        # 如果使用low bit稀疏量化，且disable_level不为L0，则使用amp自动回退
        if self.cfg.is_lowbit and self.disable_level != 'L0':
            self.cfg.use_amp = True
            self.logger.info("Using amp, automatic disabled layer set while calibrating.")
            self.cfg.amp_num = int(self.disable_level[1:])
        # 根据disable level获取自动回退层，Data-Free场景下自动回退设为L0
        elif self.calib_data:
            enable_tensor_dump = False  # 模型规模较大的时候，dump_tensor非常消耗计算、内存和存储空间，默认关闭
            try:
                self.act_states = get_features(model, self.calib_data[:5], "features.npy", enable_tensor_dump)
            except Exception as e:
                raise Exception("Please check your model and calib_data, "
                                "make sure that your model can run by model(*(calib_data[i])).", e) from e

            label_threshold_dict = self.get_label_threshold_dict()
            self.logger.info(f"The number of nn.Linear and nn.Conv2d is {len(label_threshold_dict)}.")
            auto_disable_names = self.get_auto_disable_names(label_threshold_dict)
            self.rollback_names = list(set(self.rollback_names + auto_disable_names))

            if self.disable_level != 'L0':
                self.logger.info('Automatic disabled layer names are:\n' +
                                 '\n'.join([str(name) for name in sorted(auto_disable_names)]))
        # for Data-Free
        else:
            self.logger.info("Run in Data-Free mod, disable_level set to L0")
        self.logger.info('roll back:' + '\n'.join([str(name) for name in sorted(self.rollback_names)]))

    def get_label_threshold_dict(self):
        label_threshold_dict = {}
        for key in self.act_states.keys():
            abs_max_tensor = max(abs(self.act_states[key]['t_max']), abs(self.act_states[key]['t_min']))
            range_param = abs_max_tensor / self.act_states[key]['std']
            label_threshold_dict[key] = range_param.item()
        return label_threshold_dict

    def get_auto_disable_names(self, label_threshold_dict):
        disable_level_number = int(self.disable_level[1:])
        if self.disable_level == 'L0':
            auto_disable_names = []
        else:
            sorted_label_list = list(dict(sorted(label_threshold_dict.items(),
                                                 key=lambda x: x[1],
                                                 reverse=True)).keys())
            auto_disable_names = [label for label in sorted_label_list
                                  if label not in self.rollback_names][:disable_level_number]
        return auto_disable_names

    def quantize(self, model):
        if self.model_with_accelerate:
            return self.quantize_model(model)
        else:
            return self.quantize_model(model).to(self.cfg.device)

    def enable_quant(self):
        enable_quantization(self.model, self.act_states, self.logger)

    def get_calib_data(self, calib_data):
        check_type(calib_data, list, param_name='calib_data')
        if not calib_data:
            QuantType.check_datafree_quant_type(self.cfg.model_quant_type)
            WeightQuantMethod.check_datafree_wmethod(self.cfg.w_method)
        else:
            self.check_calib_data(calib_data)
        return calib_data

    def check_calib_data(self, calib_data):
        element_not_tensor = False
        for i, calib_data_item in enumerate(calib_data):
            check_type(calib_data_item, (list, dict), param_name=f'calib_data[{i}]')
            for _, item in enumerate(calib_data_item):
                if not isinstance(item, torch.Tensor):
                    element_not_tensor = True
        if element_not_tensor:
            self.logger.warning("Not all elements in calib_data are torch.Tensor, "
                                "please make sure that your model can run by model(*(calib_data[i]))")

    def run(self, int_infer=False):
        if not isinstance(int_infer, bool):
            raise TypeError("int_infer should be bool, please check it.")

        try:
            self._run(int_infer=int_infer)
        except Exception as ex:
            raise Exception("Please check your config, model and input!", ex) from ex

    def save(self, output_path, safetensors_name=None, json_name=None, save_type=None):
        check_type(output_path, str, param_name="output_path")
        if not save_type:
            save_type = [SAVE_TYPE_NUMPY]
        check_element_type(save_type, element_type=str, value_type=list, param_name="save_type")
        if [False for save_type_item in save_type if save_type_item not in SAVE_TYPE_LIST]:
            self.logger.warning(
                f"save_type should be among choice {SAVE_TYPE_LIST} but got {save_type}, set to numpy"
            )
            save_type = [SAVE_TYPE_NUMPY]
        output_path = get_write_directory(output_path, write_mode=0o750)
        self.get_quant_params()
        if SAVE_TYPE_NUMPY in save_type:
            self.save_npy(output_path)
        if SAVE_TYPE_SAFE_TENSOR in save_type:
            if not isinstance(safetensors_name, str):
                safetensors_name = f"quant_model_weight_{self.cfg.model_quant_type.lower()}.safetensors"
                self.logger.info(f"invalid safetensors_name, set safetensors_name to default {safetensors_name}")
            if not isinstance(json_name, str):
                json_name = f"quant_model_description_{self.cfg.model_quant_type.lower()}.json"
                self.logger.info(f"invalid json_name, set json_name to default {json_name}")
            self.save_safetensor(output_path, safetensors_name, json_name)

    def save_safetensor(self, output_path, safetensors_name, json_name):
        """
        基于浮点、量化两份独立权重，存储完整的量化、浮点混合权重，用户仅需加载一个混合权重即可
        """
        quant_model_weight_path = os.path.join(output_path, safetensors_name)
        quant_model_description_path = os.path.join(output_path, json_name)
        quant_model_weight_path = get_valid_write_path(quant_model_weight_path, extensions=[".safetensors"])
        quant_model_description_path = get_valid_write_path(quant_model_description_path, extensions=[".json"])

        # 修改 safetensor 权重
        safetensor_weight = {}
        quant_model_state_dict_list = list(self.quant_param_dict.keys())
        for ori_model_state_dict_name, ori_model_state_dict in self.ori_fp_weight.items():
            # 如果浮点权重名称不在量化权重名称中，说明是浮点独有的权重，需要把浮点权重加入safetensor_weight
            if ori_model_state_dict_name not in quant_model_state_dict_list:
                # Norm 有额外的anti weight、bias，单独补充
                self.set_fp_safetensor(ori_model_state_dict_name, safetensor_weight, ori_model_state_dict)

            # 如果浮点权重名称在量化权重名称中，说明是浮点转换为量化的权重，需要把量化权重加入safetensor_weight
            else:
                self.set_quant_safetensor(ori_model_state_dict_name, safetensor_weight)

        for key, item in safetensor_weight.items():
            safetensor_weight[key] = item.cpu().contiguous()

        self.logger.info("Path of quant_model_weight.safetensors is %s ", quant_model_weight_path)
        with SafeWriteUmask(umask=0o377):
            save_file(safetensor_weight, quant_model_weight_path)
        self.logger.info("Save quant_model_weight.safetensors success!")
        self.quant_model_json_description.save(quant_model_description_path)

    def set_fp_safetensor(self, ori_model_state_dict_name, safetensor_weight, ori_model_state_dict):
        safetensor_weight[ori_model_state_dict_name] = ori_model_state_dict.clone()
        self.quant_model_json_description.change_weight_type(ori_model_state_dict_name, QuantType.FLOAT)
        if ori_model_state_dict_name in self.quantized_module_param_dict:
            for quant_param_name in self.quantized_module_param_dict[ori_model_state_dict_name]:
                safetensor_weight[quant_param_name] = self.quant_param_dict.get(quant_param_name)
                self.quant_model_json_description.change_weight_type(
                    quant_param_name, self.quant_model_json_description.model_quant_type)

    def set_quant_safetensor(self, ori_model_state_dict_name, safetensor_weight):
        safetensor_weight[ori_model_state_dict_name] = self.quant_param_dict.get(ori_model_state_dict_name)
        self.quant_model_json_description.change_weight_type(
            ori_model_state_dict_name,
            self.quant_model_json_description.model_quant_type)
        # 将该量化linear的附属参数，scale、offset 等加入safetensor_weight
        for quant_param_name in self.quantized_module_param_dict.get(ori_model_state_dict_name):
            safetensor_weight[quant_param_name] = self.quant_param_dict.get(quant_param_name)
            self.quant_model_json_description.change_weight_type(
                quant_param_name, self.quant_model_json_description.model_quant_type)

    def save_npy(self, output_path):
        quant_weight_dict = {}
        scale_dict = {}
        offset_dict = {}
        if self.cfg.model_quant_type in [QuantType.W8A8, QuantType.W8A8S]:
            deq_scale_dict = {}
            quant_bias_dict = {}
        if self.use_kvcache_quant:
            kv_cache_scale = {}
            kv_cache_offset = {}

        for name, module in self.model.named_modules():
            weight_name = name + '.weight'
            if isinstance(module, (LinearQuantizer, LinearSparseQuantizer, LowBitLinearQuantizer)):
                quant_weight_dict[name] = self.quant_param_dict.get(weight_name)
                if self.cfg.model_quant_type in [QuantType.W8A8, QuantType.W8A8S]:
                    quant_bias_dict[name] = self.quant_param_dict.get(name + '.quant_bias')
                    deq_scale_dict[name] = self.quant_param_dict.get(name + '.deq_scale')
                    scale_dict[name] = self.quant_param_dict.get(name + '.input_scale')
                    offset_dict[name] = self.quant_param_dict.get(name + '.input_offset')
                if self.cfg.model_quant_type in [QuantType.W8A16, QuantType.W4A16, QuantType.W8A8_DYNAMIC]:
                    scale_dict[name] = self.quant_param_dict.get(name + '.weight_scale')
                    offset_dict[name] = self.quant_param_dict.get(name + '.weight_offset')
            if self.use_kvcache_quant:
                self.get_kvcache_quant_params(weight_name, kv_cache_scale, kv_cache_offset)

        self.save_param(output_path, "quant_weight.npy", quant_weight_dict)
        if self.cfg.model_quant_type in [QuantType.W8A8, QuantType.W8A8S]:
            self.save_param(output_path, "input_scale.npy", scale_dict)
            self.save_param(output_path, "input_offset.npy", offset_dict)
            self.save_param(output_path, "quant_bias.npy", quant_bias_dict)
            self.save_param(output_path, "deq_scale.npy", deq_scale_dict)
        if self.cfg.model_quant_type in [QuantType.W8A16, QuantType.W4A16, QuantType.W8A8_DYNAMIC]:
            self.save_param(output_path, "weight_scale.npy", scale_dict)
            self.save_param(output_path, "weight_offset.npy", offset_dict)
        if self.use_kvcache_quant:
            self.save_param(output_path, "kv_cache_scale.npy", kv_cache_scale)
            self.save_param(output_path, "kv_cache_offset.npy", kv_cache_offset)

        anti_norm_wb = self.get_anti_fp_weight()
        if anti_norm_wb:
            self.save_param(output_path, "anti_fp_norm.npy", anti_norm_wb)

        self.logger.info("Save quant param success!")

    def save_param(self, output_path, output_name, output_file):
        output_path = os.path.join(output_path, output_name)
        self.logger.info("Path of quant param is %s ", output_path)
        output_path = get_valid_write_path(output_path)
        with SafeWriteUmask(umask=0o377):
            np.save(output_path, output_file)

    def get_anti_fp_weight(self):
        anti_norm_wb = {}
        for name, module in self.model.named_modules():
            anti_weight_name = name + '.weight'
            anti_bias_name = name + '.bias'
            if isinstance(module, NormBias):
                anti_norm_wb[anti_weight_name] = module.module.weight.cpu()
                anti_norm_wb[anti_bias_name] = module.bias.cpu()
            if isinstance(module, LlamaRMSNormBias):
                anti_norm_wb[anti_weight_name] = module.weight.cpu()
                anti_norm_wb[anti_bias_name] = module.bias.cpu()
        return anti_norm_wb

    def get_kvcache_quant_params(self, weight_name, kv_cache_scale, kv_cache_offset):
        if self.quantized_module_param_dict[weight_name]:
            for param_name in self.quantized_module_param_dict.get(weight_name):
                if 'kv_cache_scale' in param_name:
                    kv_cache_scale[param_name] = self.quant_param_dict.get(param_name)
                elif 'kv_cache_offset' in param_name:
                    kv_cache_offset[param_name] = self.quant_param_dict.get(param_name)

    def get_quant_params(self):
        self.model.eval()
        for name, module in self.model.named_modules():
            # 处理 Norm 对应的 weight、bias
            if isinstance(module, (NormBias, LlamaRMSNormBias)):
                anti_norm_weight = module.module.weight.cpu() if isinstance(module, NormBias) else module.weight.cpu()
                anti_norm_bias = module.bias.cpu()
                anti_norm_name_weight = name + '.module.weight'
                anti_norm_name_bias = name + '.module.bias'
                self.quant_param_dict[anti_norm_name_weight] = anti_norm_weight.clone().detach()
                self.quant_param_dict[anti_norm_name_bias] = anti_norm_bias.clone().detach()
                self.quantized_module_param_dict[name + '.weight'] = [anti_norm_name_weight, anti_norm_name_bias]

            # 处理Linear、以及附属scale、offset等params
            if isinstance(module, (Conv2dQuantizer, LinearQuantizer, LinearSparseQuantizer, LowBitLinearQuantizer)):
                if not module.quant_weight.is_enable:
                    continue

                quant_weight, fp_weight, weight_scale, weight_offset = self.get_param_from_quantizer(module)
                if quant_weight is None:
                    continue
                # 各种量化均需要提供 weight
                quant_weight = quant_weight.cpu()
                self.quant_param_dict[name + '.weight'] = quant_weight.to(torch.int8)

                # W4A16/W8A16 需要提供 weight_scale、weight_offset
                if self.cfg.model_quant_type in [QuantType.W8A16, QuantType.W4A16, QuantType.W8A8_DYNAMIC]:
                    self.quant_param_dict[name + '.weight_scale'] = weight_scale
                    self.quant_param_dict[name + '.weight_offset'] = weight_offset
                    self.quantized_module_param_dict[name + '.weight'].append(name + '.weight_scale')
                    self.quantized_module_param_dict[name + '.weight'].append(name + '.weight_offset')
                # W8A8/W8A8S 需要提供 deq_scale、quant_bias、input_scale、input_offset
                if self.cfg.model_quant_type in [QuantType.W8A8, QuantType.W8A8S]:
                    input_scale = module.quant_input.input_scale.cpu()
                    input_offset = module.quant_input.input_offset.cpu()
                    self.quant_param_dict[name + '.input_scale'] = input_scale
                    self.quant_param_dict[name + '.input_offset'] = input_offset
                    self.quantized_module_param_dict[name + '.weight'].append(name + '.input_scale')
                    self.quantized_module_param_dict[name + '.weight'].append(name + '.input_offset')
                    deq_scale = self.deqscale_process(input_scale, weight_scale).to(torch.float32)
                    correction = (quant_weight.to(torch.float32).sum(dim=1) * input_offset.to(torch.float32)).cpu()
                    fp_bias = self.change_bias(fp_weight, module)
                    quant_bias = torch.round(fp_bias / deq_scale - correction)
                    deq_scale = deqscale2int64_by_dtype(deq_scale, fp_weight.dtype == torch.bfloat16)

                    self.quant_param_dict[name + '.quant_bias'] = quant_bias.cpu().to(torch.int32)
                    self.quant_param_dict[name + '.deq_scale'] = deq_scale.cpu()
                    self.quantized_module_param_dict[name + '.weight'].append(name + '.quant_bias')
                    self.quantized_module_param_dict[name + '.weight'].append(name + '.deq_scale')

    def get_param_from_quantizer(self, module):
        quant_weight = None
        fp_weight, device, weight_scale, weight_offset, round_opt = self._get_module_quant_input(module)
        per_channel = False if self.cfg.mm_tensor else True
        if isinstance(module, (LinearQuantizer, Conv2dQuantizer)):
            quant_weight, _ = fake_quantize_save(fp_weight, weight_scale, weight_offset, bit=8,
                                                 round_opt=round_opt, device=device)
        if isinstance(module, LinearSparseQuantizer):
            _, _, quant_weight, _ = quant_one_weight_by_outliers(
                fp_weight, powerquant=self.cfg.nonuniform, fraction=self.cfg.fraction, num_bits=self.cfg.w_bit,
                per_channel=per_channel)
        if isinstance(module, LowBitLinearQuantizer):
            if module.disable_input:
                res = None, fp_weight, weight_scale, weight_offset
                return res
            if self.cfg.w_bit != 8:
                _, lowbit_weight_scale, lowbit_quant_weight, lowbit_weight_offset = \
                    quant_one_weight_by_outliers_low_bit(
                    fp_weight,
                    powerquant=self.cfg.nonuniform,
                    fraction=self.cfg.fraction,
                    num_bits=self.cfg.w_bit,
                    isolate_outlier_amax=False,
                    per_channel=per_channel,
                    use_cuda=False,
                    use_sigma=self.cfg.use_sigma,
                    sigma_factor=self.cfg.sigma_factor,
                    open_outlier=self.cfg.open_outlier,
                    group_size=self.cfg.group_size,
                    w_sym=self.cfg.w_sym
                )
                weight_scale = lowbit_weight_scale if weight_scale is None else weight_scale
                quant_weight = lowbit_quant_weight if quant_weight is None else quant_weight
                weight_offset = lowbit_weight_offset if weight_offset is None else weight_offset
            else:
                quant_weight, _ = fake_quantize_save(fp_weight, weight_scale, weight_offset, bit=8,
                                                     round_opt=round_opt, device=device)
        res = quant_weight, fp_weight, weight_scale, weight_offset
        return res

    def deqscale_process(self, input_scale, scale):
        deq_scale = input_scale * scale
        if deq_scale.ndim > 1:
            deq_scale = deq_scale.squeeze(1)
        deq_scale = deq_scale.cpu()
        return deq_scale

    def change_bias(self, fp_weight, module):
        if module.bias is None:
            bias_shape = fp_weight.shape[0]
            fp_bias = torch.zeros(bias_shape)
        else:
            fp_bias = module.bias.cpu()
        return fp_bias

    def deepcopy_model(self, model: nn.Module, model_with_accelerate=True):

        # 原模型转移到CPU上
        for mod in model.modules():
            try:
                # npu, cuda -> cpu
                mod.cpu()
            except Exception as e:
                # meta -> cpu
                self.logger.info("transferring meta to cpu", e)
                if hasattr(mod, HF_HOOK):
                    mod._hf_hook.detach_hook(mod)

        # 深拷贝model
        new_model = copy.deepcopy(model)

        # 删除accelerate封装的forward函数，将备份的forward函数恢复
        new_model = remove_hook_from_module(new_model, True)

        # 将原模型的权重恢复到NPU、CUDA（或meta）上
        if model_with_accelerate:
            for mod in model.modules():
                if hasattr(mod, HF_HOOK):
                    mod._hf_hook.init_hook(mod)
        else:
            model.to(self.device_org)

        return new_model

    def run_calib_mode(self):
        amp_done = not self.cfg.use_amp
        for data in tqdm(self.calib_data):
            if isinstance(data, tuple) or isinstance(data, list):
                self.model(*data)
            elif isinstance(data, dict):
                self.model(**data)

            if not amp_done:
                self.run_amp()
                amp_done = True

    def run_amp(self):
        max_input_dict = {}

        for name, module in self.model.named_modules():
            if isinstance(module, LowBitLinearQuantizer):
                max_input_dict[name] = module.max_input_num
        layers_to_disable = list(
            dict(sorted(max_input_dict.items(),
                        key=lambda x: x[1], reverse=True)).keys()
        )[:self.cfg.amp_num]
        for name, module in self.model.named_modules():
            if name in layers_to_disable:
                module.disable_input = True
                module.disable_quant_weight()
        self.rollback_names.extend(layers_to_disable)
        self.logger.info('roll back:' + '\n'.join([str(name) for name in sorted(self.rollback_names)]))

    def run_datafree_mode(self):
        for name, module in self.model.named_modules():
            if isinstance(module, (Conv2dQuantizer, LinearQuantizer)):
                self.logger.info(f"Run Data-Free, quantize linear name: {name}")
                module.quant_weight(module.weight)

    def quantize_model(self, model):
        def _set_module(ori_mod, submodule_key, module):
            tokens = submodule_key.split('.')
            sub_tokens = tokens[:-1]
            cur_mod = ori_mod
            for s in sub_tokens:
                cur_mod = getattr(cur_mod, s)
            setattr(cur_mod, tokens[-1], module)

        if self.cfg.do_smooth:
            dummy_input = torch.randint(0, 100, (1, 128)).type(torch.int64)
            norm_class = self.get_norm_class(model, norm_class_name=self.norm_class_name)
            dag = TorchDAGAdapter(model, dummy_input, hook_nodes=norm_class)
            list_infos = dag.get_llm_network_pattern_auto()
            attn_list, mhsa_ln_list, ffn_list, ffn_ln_list = list_infos[0], list_infos[2], list_infos[3], list_infos[4]
            
            if not (len(attn_list) == len(mhsa_ln_list) == len(ffn_list) == len(ffn_ln_list)):
                raise ValueError("get network pattern by dag error")
            for _, (attn, mhsa_ln, ffn, ffn_ln) in enumerate(zip(attn_list, mhsa_ln_list, ffn_list, ffn_ln_list)):
                modules = [attn, mhsa_ln, ffn, ffn_ln]
                cur_decoder, cur_decoder_type, modules_info, cur_decoder_name = \
                    self.split_module_name_get_info(modules, model, self.cfg.down_proj_type)
                decoder_info = {
                    'attn': modules_info[0],
                    'mlp': modules_info[2],
                    'ln1': modules_info[1],
                    'ln2': modules_info[3],
                    'qkv': modules_info[4],
                    'ffn': modules_info[5]
                }
                quant_mod = QuantXDecoderLayer(cur_decoder, self.cfg, cur_decoder_name, decoder_info)
                if hasattr(cur_decoder, HF_HOOK):
                    add_hook_to_module(quant_mod, cur_decoder._hf_hook)
                    remove_hook_from_module(cur_decoder)
                _set_module(model, cur_decoder_name, quant_mod)

        for name, mod in model.named_modules():
            if name in self.rollback_names:
                continue
            if isinstance(mod, nn.Conv2d):
                quant_mod = Conv2dQuantizer(cfg=self.cfg, logger=self.logger)
                quant_mod.set_param(mod)

                # 拷贝accelerate定义的hook
                if hasattr(mod, HF_HOOK):
                    add_hook_to_module(quant_mod, mod._hf_hook)
                    remove_hook_from_module(mod)

                _set_module(model, name, quant_mod)
                del mod
            elif isinstance(mod, nn.Linear) or isinstance(mod, nn.modules.linear.NonDynamicallyQuantizableLinear):
                if self.cfg.is_lowbit:
                    quant_mod = LowBitLinearQuantizer(cfg=self.cfg, logger=self.logger, name=name)
                elif self.cfg.model_quant_type is not QuantType.W8A8S:
                    quant_mod = LinearQuantizer(cfg=self.cfg, logger=self.logger)
                else :
                    quant_mod = LinearSparseQuantizer(cfg=self.cfg, logger=self.logger)
                quant_mod.set_param(mod)

                # 拷贝accelerate定义的hook
                if hasattr(mod, HF_HOOK):
                    add_hook_to_module(quant_mod, mod._hf_hook)
                    remove_hook_from_module(mod)

                _set_module(model, name, quant_mod)
                del mod
        gc.collect()

        return model

    # 获取decoder相关信息
    def split_module_name_get_info(self, module_names, model, down_proj_type):
        item0 = module_names[0][0] if isinstance(module_names[0], list) else module_names[0]
        item2 = module_names[2][0] if isinstance(module_names[2], list) else module_names[2]
        module_names_ = [item0, module_names[1], item2, module_names[3]]

        qkv_item = module_names[0] if isinstance(module_names[0], list) else [module_names[0]]
        ffn_item = module_names[2] if isinstance(module_names[2], list) else [module_names[2]]

        qkv_tokens = [item.split('.') for item in qkv_item]
        qkv_list = [item[-1] for item in qkv_tokens]

        ffn_tokens = [item.split('.') for item in ffn_item]
        ffn_list = [item[-1] for item in ffn_tokens if item[-1] not in down_proj_type]


        tokens = [module_names_[i].split('.') for i in range(len(module_names_))]
        cur_mod = model
        layername = ''
        for i, item in enumerate(tokens[0]):
            layername += item + '.'
            cur_mod = getattr(cur_mod, item)
            if item.isnumeric():
                module_infos = [token[i+1] for token in tokens]
                module_infos.append(qkv_list)
                module_infos.append(ffn_list)
                break
        mod_type = cur_mod.__class__.__name__

        ret = cur_mod, mod_type, module_infos, layername[:-1]
        return ret

    def get_norm_class(self, model, norm_class_name=None):
        if norm_class_name is not None:
            norm_class = list(set([m.__class__ for m in model.modules() \
                if norm_class_name.lower() in m.__class__.__name__.lower()])) 
        else:
            norm_class = list(set([m.__class__ for m in model.modules() if "norm" in m.__class__.__name__.lower()]))
            if len(norm_class) != 1:
                raise ValueError("Customized norm classes aren't detected, please manually enter the norm class.")
        return norm_class

    def _get_module_quant_input(self, module):
        fp_weight = module.weight.cpu()
        weight_scale, weight_offset = module.quant_weight.weight_scale, module.quant_weight.weight_offset
        device = weight_scale.device if weight_scale is not None else None
        scale = weight_scale.cpu() if weight_scale is not None else None
        offset = weight_offset.cpu() if weight_offset is not None else None
        round_opt = False if isinstance(module, LowBitLinearQuantizer) else module.quant_weight.round_opt
        ret = fp_weight, device, scale, offset, round_opt
        return ret

    def _run(self, calib_amp=5, int_infer=False):
        if not isinstance(calib_amp, int) or calib_amp < 1:
            raise TypeError("calib_amp should be int and more than 0 and not more than length of calib data,"
                            " please check it.")

        self.logger.info("Calibration start!")
        self.model.eval()
        if self.calib_data:
            if self.cfg.calib_mode == 0 and self.cfg.is_dynamic:
                self.run_datafree_mode()
            elif self.cfg.calib_mode == 0 and not self.cfg.is_dynamic:
                with torch.no_grad():
                    self.run_calib_mode()
            elif self.cfg.calib_mode == 1:
                try:
                    layer_wise_calib(self.model, self.all_tensors, self.cfg.device)
                except Exception as e:
                    raise Exception("Please check your model, all_tensors and config", e) from e
                del self.all_tensors
            else:
                raise ValueError("Calibration mode not supported!")
        else:
            self.run_datafree_mode()

        self.logger.info("Calibration end!")

        disable_calibration(self.model, self.logger)
        if self.cfg.a_bit != 8 and int_infer:
            self.logger.info("int_infer works only in W8A8 case.")
            int_infer = False
        if int_infer:
            enable_int_infer(self.model, self.logger)


def load_backup_model(backup_model: nn.Module, device=None, model_with_accelerate: bool = False):
    # 如果原model带有accelerate的hook，则这些hook也会被拷贝并且这些hook指向的还是model而非backup_model
    # 1. 修改hook的指向
    # 2. 将backup_model的权重放在正确的设备上

    if model_with_accelerate:
        for mod in backup_model.modules():
            # 拷贝accelerate定义的hook
            if hasattr(mod, HF_HOOK):
                add_hook_to_module(mod, mod._hf_hook)
    else:
        backup_model.to(device)

    return backup_model


def enable_quantization(model, act_states, logger=None):
    _ = logger  # Bypassing not using
    for name, module in model.named_modules():
        if isinstance(module, Quantizer):
            states_name = ".".join(name.split(".")[:-1])
            abs_max_tensor = max(abs(act_states[states_name]["t_max"]), abs(act_states[states_name]["t_min"]))
            range_param = abs_max_tensor / act_states[states_name]["std"]
            module.enable_quantization(name, range_param)
            if module.is_input:
                module.init_act_and_observer(module.cfg)
        if isinstance(module, LowBitQuantizer):
            module.enable_quantization()
            if module.is_input:
                module.init_act_and_observer(module.cfg)
        if isinstance(module, QuantXDecoderLayer):
            module.calibration = True


def disable_calibration(model, logger=None):
    _ = logger  # Bypassing not using
    for module in model.modules():
        if isinstance(module, Quantizer):
            module.disable_calib()


def enable_calibration(model, logger=None):
    _ = logger  # Bypassing not using
    for module in model.modules():
        if isinstance(module, Quantizer):
            module.enable_calib()


def disable_int_infer(model, logger=None):
    _ = logger  # Bypassing not using
    for module in model.modules():
        if isinstance(module, Quantizer):
            module.disable_int_infer()


def enable_int_infer(model, logger=None):
    _ = logger  # Bypassing not using
    for module in model.modules():
        if isinstance(module, Quantizer):
            module.enable_int_infer()


def set_ratio(model, ratio=0.9, logger=None):
    for name, module in model.named_modules():
        if isinstance(module, Quantizer):
            if logger:
                logger.info('Set the ratio: %s', name)
            module.set_ratio(ratio)


def judge_model_with_accelerate(model: nn.Module):
    for _, mod in model.named_modules():
        if hasattr(mod, HF_HOOK):
            return True
    return False


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