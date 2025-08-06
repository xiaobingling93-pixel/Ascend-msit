#  -*- coding: utf-8 -*-
#  Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#  http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import torch

from msmodelslim.core.QAL import QStorage
from msmodelslim.core.QAL.qbase import QDType, QScope, QParam, QScheme
from msmodelslim.core.QAL.qregistry import QFuncRegistry
from msmodelslim.utils.exception import SchemaValidateError


@QFuncRegistry.register(dispatch_key=(QDType.INT8, QScope.PER_TOKEN, False), api_name="calculate_qparam")
@QFuncRegistry.register(dispatch_key=(QDType.INT8, QScope.PER_TOKEN, True), api_name="calculate_qparam")
@QFuncRegistry.register(dispatch_key=(QDType.INT8, QScope.PER_CHANNEL, False), api_name="calculate_qparam")
@QFuncRegistry.register(dispatch_key=(QDType.INT8, QScope.PER_CHANNEL, True), api_name="calculate_qparam")
@QFuncRegistry.register(dispatch_key=(QDType.INT8, QScope.PER_TENSOR, False), api_name="calculate_qparam")
@QFuncRegistry.register(dispatch_key=(QDType.INT8, QScope.PER_TENSOR, True), api_name="calculate_qparam")
def int8_param(
        min_val: torch.Tensor,
        max_val: torch.Tensor,
        q_dtype: QDType,
        q_scope: QScope,
        symmetric: bool,
        **kwargs
) -> QParam:
    eps = torch.tensor([torch.finfo(torch.float32).eps]).type_as(min_val)
    min_val = torch.min(min_val, torch.zeros_like(min_val))
    max_val = torch.max(max_val, torch.zeros_like(max_val))
    q_signed = kwargs.get("q_signed", True)
    integral_zero_point = kwargs.get("integral_zero_point", True)
    max_bound = kwargs.get("max_bound", None)  # for w4a8   max_bound=119

    if not symmetric:
        max_bound = 2 ** 8 - 1 if max_bound is None else max_bound
        # asymmetric quantization
        scale = (max_val - min_val) / max_bound
        scale = torch.max(scale, eps)
        offset = -1 * min_val / scale
        if integral_zero_point:
            if isinstance(offset, torch.Tensor):
                offset = offset.round()
            else:
                offset = float(round(offset))

        if q_signed:
            qmin = -1 * (2 ** (8 - 1))
            offset += qmin

    else:
        # symmetric quantization
        max_bound = 2 ** (8 - 1) - 1 if max_bound is None else max_bound
        max_val_pos = torch.max(-min_val, max_val)
        scale = max_val_pos / float(max_bound)
        scale = torch.max(scale, eps)
        offset = torch.tensor(0.0).type_as(min_val).expand(scale.shape)

    return QParam(
        scheme=QScheme(
            dtype=q_dtype,
            scope=q_scope,
            symmetric=symmetric,
        ),
        ext={
            "scale": scale,
            "offset": offset
        }
    )


@QFuncRegistry.register(dispatch_key=(QDType.INT8, QScope.PER_GROUP, False), api_name="calculate_qparam")
@QFuncRegistry.register(dispatch_key=(QDType.INT8, QScope.PER_GROUP, True), api_name="calculate_qparam")
def int8_per_group_param(
        min_val: torch.Tensor,
        max_val: torch.Tensor,
        q_dtype: QDType,
        q_scope: QScope,
        symmetric: bool,
        **kwargs
) -> QParam:
    group_size = min_val.shape[-1]
    q_param = int8_param(min_val, max_val, q_dtype, q_scope, symmetric, **kwargs)
    q_param.ext['group_size'] = group_size
    return q_param


@QFuncRegistry.register(dispatch_key=(QDType.FLOAT, QDType.INT8, QScope.PER_CHANNEL, True), api_name="quantize")
@QFuncRegistry.register(dispatch_key=(QDType.FLOAT, QDType.INT8, QScope.PER_CHANNEL, False), api_name="quantize")
@QFuncRegistry.register(dispatch_key=(QDType.FLOAT, QDType.INT8, QScope.PER_TOKEN, False), api_name="quantize")
@QFuncRegistry.register(dispatch_key=(QDType.FLOAT, QDType.INT8, QScope.PER_TOKEN, True), api_name="quantize")
@QFuncRegistry.register(dispatch_key=(QDType.FLOAT, QDType.INT8, QScope.PER_TENSOR, False), api_name="quantize")
@QFuncRegistry.register(dispatch_key=(QDType.FLOAT, QDType.INT8, QScope.PER_TENSOR, True), api_name="quantize")
def int8_quantize(tensor: QStorage, q_param: QParam) -> QStorage:
    scale = q_param.ext["scale"]
    offset = q_param.ext["offset"] if "offset" in q_param.ext else torch.zeros_like(scale)
    inplace = q_param.ext.get("inplace", False)
    max_bound = q_param.ext.get("max_bound", False)
    input_tensor = tensor.value
    if inplace:
        input_tensor = input_tensor.div_(scale).add_(offset).round_()
    else:
        input_tensor = (input_tensor / scale + offset).round_()
    if max_bound:
        input_tensor = input_tensor.clamp_(min=-max_bound, max=max_bound)
    return tensor.same_like(input_tensor).to(QDType.INT8)


@QFuncRegistry.register(dispatch_key=(QDType.INT8, QDType.INT8, QScope.PER_CHANNEL, True), api_name="dequantize")
@QFuncRegistry.register(dispatch_key=(QDType.INT8, QDType.INT8, QScope.PER_CHANNEL, False), api_name="dequantize")
@QFuncRegistry.register(dispatch_key=(QDType.INT8, QDType.INT8, QScope.PER_TOKEN, False), api_name="dequantize")
@QFuncRegistry.register(dispatch_key=(QDType.INT8, QDType.INT8, QScope.PER_TOKEN, True), api_name="dequantize")
@QFuncRegistry.register(dispatch_key=(QDType.INT8, QDType.INT8, QScope.PER_TENSOR, False), api_name="dequantize")
@QFuncRegistry.register(dispatch_key=(QDType.INT8, QDType.INT8, QScope.PER_TENSOR, True), api_name="dequantize")
def int8_dequantize(tensor: QStorage, q_param: QParam) -> QStorage:
    scale = q_param.ext["scale"]
    offset = q_param.ext["offset"] if "offset" in q_param.ext else torch.zeros_like(scale)
    inplace = q_param.ext.get("inplace", False)
    input_tensor = tensor.value
    if inplace:
        input_tensor = input_tensor.sub_(offset).mul_(scale)
    else:
        input_tensor = (input_tensor - offset) * scale
    return tensor.same_like(input_tensor).to(QDType.FLOAT)


@QFuncRegistry.register(dispatch_key=(QDType.FLOAT, QDType.INT8, QScope.PER_GROUP, False), api_name="quantize")
@QFuncRegistry.register(dispatch_key=(QDType.FLOAT, QDType.INT8, QScope.PER_GROUP, True), api_name="quantize")
def int8_per_group_quantize(tensor: QStorage, q_param: QParam) -> QStorage:
    group_size = q_param.ext.get("group_size", -1)
    if group_size < 0:
        raise SchemaValidateError(f"group quantize group_size must be greater than 0 but got group_size = {group_size}",
                                  action=f"Please make sure group_size is greater than 0")
    org_shape = tensor.value.shape
    tensor.value = tensor.value.reshape(-1, group_size)  # reshape for per_group
    tensor = int8_quantize(tensor, q_param)
    tensor.value = tensor.value.reshape(org_shape)  # reshape back
    return tensor


@QFuncRegistry.register(dispatch_key=(QDType.INT8, QDType.INT8, QScope.PER_GROUP, False), api_name="dequantize")
@QFuncRegistry.register(dispatch_key=(QDType.INT8, QDType.INT8, QScope.PER_GROUP, True), api_name="dequantize")
def int8_per_group_dequantize(tensor: QStorage, q_param: QParam) -> QStorage:
    group_size = q_param.ext.get("group_size", -1)
    if group_size < 0:
        raise SchemaValidateError(f"group quantize group_size must be greater than 0 but got group_size = {group_size}",
                                  action=f"Please make sure group_size is greater than 0")
    org_shape = tensor.value.shape
    tensor.value = tensor.value.reshape(-1, group_size)  # reshape for per_group
    tensor = int8_dequantize(tensor, q_param)
    tensor.value = tensor.value.reshape(org_shape)  # reshape back
    return tensor
