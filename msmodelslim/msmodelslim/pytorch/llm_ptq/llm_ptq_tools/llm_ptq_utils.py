# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.

from enum import Enum
from safetensors.torch import load_file
import torch
from ascend_utils.common.security import json_safe_dump
from ascend_utils.common.security import json_safe_load
from ascend_utils.common.security import get_valid_read_path
from ascend_utils.common.security import MAX_READ_FILE_SIZE_512G
from msmodelslim import logger

SAVE_TYPE_NUMPY = "numpy"
SAVE_TYPE_SAFE_TENSOR = "safe_tensor"
SAVE_TYPE_ASCENDV1 = "ascendV1"
SAVE_TYPE_LIST = [SAVE_TYPE_NUMPY, SAVE_TYPE_SAFE_TENSOR, SAVE_TYPE_ASCENDV1]


class QuantType(str, Enum):
    UNKNOWN = "UNKNOWN"  # 未被识别的类型
    W8A16 = "W8A16"  # W8A16量化，Matmul的weight为8bit，activation为16bit
    W4A16 = "W4A16"  # W4A16量化，Matmul的weight为4bit，activation为16bit
    W8A8 = "W8A8"  # W8A8量化，Matmul的weight、activation均为8bit
    W8A8S = "W8A8S"  # 稀疏量化，Matmul的weight、activation均为8bit，且weight经过稀疏(权重数值分布范围可能小于8bit)
    W8A8SC = "W8A8SC"  # 稀疏量化压缩后的权重
    FLOAT = "FLOAT"  # 浮点
    KV8 = "C8"  # kvcache量化，kvcache为8bit
    FAQuant = "FAQuant" # flashattention量化为8bit
    NF4 = "NF4"  # Normal Float 4-Bit量化
    W8A8_DYNAMIC = "W8A8_DYNAMIC"  # W8A8静态量化与per-token动态量化混合量化

    W4A8_DYNAMIC = "W4A8_DYNAMIC"  # W4A8静态量化与per-token动态量化混合量化

    W8A8_TIMESTEP = "W8A8_TIMESTEP"  # 分时间步量化

    @staticmethod
    def get_quant_type(params):
        w_bit = params['w_bit']
        a_bit = params['a_bit']
        w_method = params['w_method']
        is_sparse = params['is_sparse']
        is_dynamic = params['is_dynamic']
        is_lowbit = params['is_lowbit']

        if w_bit == 8 and a_bit == 8 and params['is_timestep_quant']:
            return QuantType.W8A8_TIMESTEP
        
        if is_dynamic:
            return QuantType.get_dynamic_quant_type(w_bit, a_bit)
        if is_sparse:
            return QuantType.W8A8S
        if w_bit == 8 and a_bit == 8:
            return QuantType.W8A8
        if w_bit == 8 and a_bit == 16:
            return QuantType.W8A16
        if w_bit == 4 and a_bit == 16 and w_method in QuantType.NF4:
            return QuantType.NF4
        if w_bit == 4 and a_bit == 16:
            return QuantType.W4A16
        if w_bit == 16 and a_bit == 16:
            return QuantType.FLOAT
        if is_lowbit and w_bit == 4 and a_bit == 8:
            return QuantType.W8A8S
        return QuantType.UNKNOWN

    @staticmethod
    def is_value_in_enum(quant_type_value):
        return quant_type_value in ["UNKNOWN", "W8A16", "W4A16", "W8A8", "W8A8S", 
                                    "W8A8SC", "FLOAT", "W8A8_DYNAMIC", "NF4"]

    @staticmethod
    def check_instance_of_enum(instance):
        if not isinstance(instance, QuantType):
            raise TypeError("please check QuantType")

    @staticmethod
    def check_datafree_quant_type(quant_type_value):
        if quant_type_value not in [QuantType.W8A16, QuantType.W4A16, 
                                    QuantType.W8A8_DYNAMIC, QuantType.NF4, QuantType.W4A8_DYNAMIC]:
            raise ValueError(f"QuantType.{quant_type_value} does not support \
                             Data-Free, please check your QuantConfig.")

    @staticmethod
    def get_dynamic_quant_type(w_bit, a_bit):
        if w_bit == 8 and a_bit == 8:
            return QuantType.W8A8_DYNAMIC
        if w_bit == 4 and a_bit == 8:
            return QuantType.W4A8_DYNAMIC
        return QuantType.UNKNOWN


class QuantModelJsonDescription:
    model_quant_type_name = "model_quant_type"
    kv_cache_type_name = "kv_cache_type"
    fa_quant_type_name = "fa_quant_type"
    version_type_name = "version"
    group_size_name = "group_size"
    kv_quant_type_name = "kv_quant_type"

    def __init__(self, model_quant_type, use_kvcache_quant=False, use_fa_quant=False, version_name=None, group_size=0):
        self.quant_model_description = {}

        self.check_version_format(version_name)
        self.change_version_name(version_name)

        QuantType.check_instance_of_enum(model_quant_type)
        self.model_quant_type = model_quant_type
        self.change_model_type(model_quant_type)
        if use_kvcache_quant and use_fa_quant:
            raise ValueError("KV-cache and FA cannot be quantized at the same time!")
        self.change_kvcache_type(use_kvcache_quant)
        self.change_fa_quant_type(use_fa_quant)
        self.change_group_size(group_size)

    @staticmethod
    def check_description(quant_model_json_description=None, quant_model_json_description_path=None):
        """
        校验量化json描述是否正常，仅对key、value的数据类型进行校验
        校验json实例对象、或path所对应json
        """
        json_description = None
        if quant_model_json_description_path:
            json_description = json_safe_load(quant_model_json_description_path)
        elif quant_model_json_description:
            json_description = quant_model_json_description
        else:
            raise TypeError("please provide quant_model_json_description or quant_model_json_description_path.")

        if not isinstance(json_description, dict):
            raise TypeError("quant_model_json_description must be a dict.")
        for weight_name, weight_type in json_description.items():
            if not isinstance(weight_name, str):
                raise TypeError("weight name in quant_model_json_description must be str.")
            if not QuantType.is_value_in_enum(weight_type):
                raise TypeError("weight type in quant_model_json_description must be QuantType.")
        json_keys = json_description.keys()
        if len(json_keys) == 0:
            raise ValueError("quant_model_json_description does not contain any data.")
        if list(json_keys)[0] != QuantModelJsonDescription.model_quant_type_name:
            raise ValueError("quant_model_json_description must have model quant type.")
        return json_description

    @staticmethod
    def check_safetensor(quant_model_safetensor=None, quant_model_safetensor_path=None):
        """
        校验safetensor权重是否正常，仅对key、value的数据类型进行校验
        校验safetensor实例对象、或path所对应safetensor
        """
        safetensor_weight = None
        if quant_model_safetensor_path:
            try:
                quant_model_safetensor_path = get_valid_read_path(
                    quant_model_safetensor_path,
                    extensions=".safetensors",
                    size_max=MAX_READ_FILE_SIZE_512G)
                safetensor_weight = load_file(quant_model_safetensor_path)
            except Exception as ex:
                logger.error('please check your safetensor file. %s', str(ex))
                raise ex
        elif quant_model_safetensor:
            safetensor_weight = quant_model_safetensor
        else:
            raise TypeError("please provide quant_model_safetensor or quant_model_safetensor_path.")
        if not isinstance(safetensor_weight, dict):
            raise TypeError("quant_model_safetensor must be a dict.")
        for weight_name, weight in safetensor_weight.items():
            if not isinstance(weight_name, str):
                raise TypeError("weight name in safetensor must be str.")
            if not isinstance(weight, torch.Tensor):
                raise TypeError("weight safetensor must be tensor.")
        return safetensor_weight

    @staticmethod
    def check_description_match(quant_model_json_description=None, quant_model_json_description_path=None,
                                quant_model_safetensor=None, quant_model_safetensor_path=None):
        """
        校验量化json描述和量化safetensor是否匹配：json和safetensor的key是否相同，并校验json和safetensor的key、value数据类型
        json描述、safetensor权重可以是实例对象，也可以是路径
        """
        json_description = QuantModelJsonDescription.check_description(
            quant_model_json_description, quant_model_json_description_path)
        safetensor_weight = QuantModelJsonDescription.check_safetensor(
            quant_model_safetensor, quant_model_safetensor_path)
        json_keys = list(json_description.keys())
        safetensor_keys = list(safetensor_weight.keys())
        json_keys = json_keys[1:]
        # safetensor保存后，读取出来的key顺序和保存时的有区别
        if set(json_keys) != set(safetensor_keys):
            raise ValueError("quant_model_json_description and quant_model_safetensor do not match.")

    
    @staticmethod
    def check_version_format(version_name=None):
        """
        校验version_name格式是否为x.x.x
        @param version_name: str, version_name
        @return: None
        """
        if version_name is None:
            return
        
        import re
        pattern = r'^[1-9]\d*\.\d+\.\d+$'
        if not bool(re.fullmatch(pattern, version_name)):
            raise ValueError(f'version_name format must be x.x.x, current is {version_name}')


    def change_version_name(self, version_name=None):
        """
        修改version_name
        @param version_name: str, version_name
        @return: None
        """
        if version_name is None:
            return
        self.quant_model_description[QuantModelJsonDescription.version_type_name] = version_name
    
    def change_group_size(self, group_size=0):
        if group_size <= 0:
            group_size = 0
        self.quant_model_description[QuantModelJsonDescription.group_size_name] = group_size

    
    def change_model_type(self, model_quant_type):
        QuantType.check_instance_of_enum(model_quant_type)
        self.quant_model_description[QuantModelJsonDescription.model_quant_type_name] = model_quant_type

    def change_kvcache_type(self, use_kvcache_quant):
        if use_kvcache_quant:
            self.quant_model_description[QuantModelJsonDescription.kv_cache_type_name] = QuantType.KV8
            self.quant_model_description[QuantModelJsonDescription.kv_quant_type_name] = QuantType.KV8

    def change_fa_quant_type(self, use_fa_quant):        
        if use_fa_quant:
            self.quant_model_description[QuantModelJsonDescription.fa_quant_type_name] = QuantType.FAQuant

    def change_weight_type(self, weight_name, weight_quant_type):
        QuantType.check_instance_of_enum(weight_quant_type)
        self.quant_model_description[weight_name] = weight_quant_type

    def remove_weight(self, weight_name):
        self.quant_model_description.pop(weight_name, None)

    def save(self, path):
        logger.info("Path of quant_model_description_json is %r ", path)
        json_safe_dump(self.quant_model_description, path, indent=2)
        logger.info("Save quant_model_description_json success!")


class WeightQuantMethod(str, Enum):
    MinMax = 'MinMax'
    GPTQ = 'GPTQ'
    HQQ = 'HQQ'
    NF = 'NF'

    @staticmethod
    def get_wmethod_config(w_method):
        w_hessian = False
        hqq = False
        if w_method == WeightQuantMethod.MinMax:
            pass
        elif w_method == WeightQuantMethod.GPTQ:
            w_hessian = True
        elif w_method == WeightQuantMethod.HQQ:
            hqq = True
        elif w_method == WeightQuantMethod.NF:
            pass
        else:
            raise ValueError(f"w_method {w_method} illegal, please check it.")
        return w_hessian, hqq

    @staticmethod
    def check_quant_type(quant_type, w_method):
        if quant_type in [QuantType.W8A8, QuantType.W8A8S] and w_method != WeightQuantMethod.MinMax:
            raise ValueError(f"w_method {w_method} does not support quant_type {quant_type}, please check it.")

    @staticmethod
    def check_datafree_wmethod(w_method):
        if w_method not in [WeightQuantMethod.MinMax, WeightQuantMethod.HQQ, WeightQuantMethod.NF]:
            raise ValueError(f"w_method {w_method} does not support Data-Free, please check it.")
