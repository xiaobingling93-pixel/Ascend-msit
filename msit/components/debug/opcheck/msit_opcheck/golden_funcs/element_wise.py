# -*- coding: utf-8 -*-
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

import numpy
import tensorflow as tf
import numpy as np
import torch
from scipy import special

from msit_opcheck.graph_parser import OpInfo
from msit_opcheck.conversion.shape_convert import transform
from msit_opcheck.conversion.dtype_convert import bfloat16_conversion, numpy_to_torch_tensor


def str_to_dtype(dtype_str):
    dtype_map = {
        'float16': torch.float16,
        'float32': torch.float32,
        'float64': torch.float64,
        'int8': torch.int8,
        'int16': torch.int16,
        'int32': torch.int32,
        'int64': torch.int64,
        'uint8': torch.uint8,
        'bfloat16': torch.bfloat16,
        'bool': torch.bool
    }
    if dtype_str in dtype_map:
        return dtype_map[dtype_str]
    else:
        raise ValueError('Unsupported dtype: {}'.format(dtype_str))


def due_fp16_overflow(data):
    """Overflow interception"""
    data = np.maximum(data, -65504)
    data = np.minimum(data, 65504)
    data = np.nan_to_num(data)
    return data


def _broadcast_to(context: OpInfo):
    tf.compat.v1.disable_eager_execution()

    input_x = context.param.get("input_arrays")[0]
    input_x_shape = input_x.shape
    input_shape = context.param.get("shape")

    len_diff = len(input_shape) - len(input_x_shape)
    input_x_shape = (1,) * len_diff + input_x_shape
    brc_shape = [input_x_shape[index] if dim_value == -1 else dim_value for index, dim_value in enumerate(input_shape)]

    input_x_holder = tf.placeholder(input_x.dtype, shape=input_x.shape)
    input_shape_holder = tf.Variable(brc_shape)
    out = tf.broadcast_to(input_x_holder, input_shape_holder)

    with tf.compat.v1.Session()  as sess:
        result = sess.run(out, feed_dict={input_x_holder: input_x, input_shape_holder: brc_shape})
    return [result]


def _mul(context: OpInfo):
    x1 = context.param.get("input_arrays")[0]
    x2 = context.param.get("input_arrays")[1]
    output_dtype = context.param.get("output_dtypes")
    input_dtype = context.param.get("stc_input_dtypes")
    if "complex32" in context.param.get("stc_input_dtypes"):
        xreal, ximag = numpy.split(x1, 2, axis=-1)
        yreal, yimag = numpy.split(x2, 2, axis=-1)
        zreal = xreal * yreal - ximag * yimag
        zimag = xreal * yimag + ximag * yreal
        res = numpy.concatenate((zreal, zimag), axis=-1)
        return res
    elif "bool" in context.param.get("stc_input_dtypes"):
        x1 = torch.tensor(x1)
        x2 = torch.tensor(x2)
        res = torch.mul(x1, x2).detach().numpy()
        return res
    else:
        if input_dtype[0] == "bfloat16":
            x1 = x1.astype("float32")
            x2 = x2.astype("float32")
        if input_dtype[0]!=input_dtype[1] :
            x1 = x1.astype("float32")
            x2 = x2.astype("float32")
        tensor_x1 = tf.compat.v1.placeholder(x1.dtype, shape=x1.shape)
        tensor_x2 = tf.compat.v1.placeholder(x2.dtype, shape=x2.shape)
        feed_dict = {tensor_x1: x1, tensor_x2: x2}
        out = tf.multiply(tensor_x1, tensor_x2)
        with tf.compat.v1.Session() as sess:
            res = sess.run(out, feed_dict=feed_dict)

        output_dtype = bfloat16_conversion(output_dtype)
        res = res.astype(output_dtype[0])
        return [res]


def _abs(context: OpInfo):
    x = context.param.get("input_arrays")[0]
    if "complex32" in context.param.get("stc_input_dtypes"):
        real, imag = numpy.split(x, 2, axis=-1)
        res = numpy.sqrt(real * real + imag * imag)
        return res
    elif "complex64" in context.param.get("stc_input_dtypes"):
        return numpy.abs(x)
    return tf.raw_ops.ComplexAbs(x)


def _atan(context: OpInfo):
    x1 = context.param.get("input_arrays")[0]
    x_dtype = x1.dtype
    if "bfloat16" in str(x_dtype) or "float16" in str(x_dtype):
        x1 = x1.astype("float32")
    res = numpy.arctan(x1)
    output_dtypes = bfloat16_conversion(context.param.get("output_dtypes"))
    return res.astype(output_dtypes[0])


def _atan2(context: OpInfo):
    x1, x2 = context.param.get("input_arrays")
    x_dtype = x1.dtype
    if "bfloat16" in str(x_dtype) or "float16" in str(x_dtype):
        x1 = x1.astype("float32")
        x2 = x2.astype("float32")
    res = numpy.arctan2(x1, x2)
    output_dtypes = bfloat16_conversion(context.param.get("output_dtypes"))
    return res.astype(output_dtypes[0])


def _atanh(context: OpInfo):
    x1 = context.param.get("input_arrays")[0]
    x_dtype = x1.dtype
    if "bfloat16" in str(x_dtype) or "float16" in str(x_dtype):
        x1 = x1.astype("float32")
    res = numpy.arctanh(x1)
    output_dtypes = bfloat16_conversion(context.param.get("output_dtypes"))
    return res.astype(output_dtypes[0])
    

def _ceil(context: OpInfo):
    x1 = context.param.get("input_arrays")[0]
    x_dtype = x1.dtype
    if "bfloat16" in str(x_dtype) or "float16" in str(x_dtype):
        x1 = x1.astype("float32")
    res = numpy.ceil(x1)
    output_dtypes = bfloat16_conversion(context.param.get("output_dtypes"))
    return res.astype(output_dtypes[0])


def _expm1(context: OpInfo):
    x1 = context.param.get("input_arrays")[0]
    x_dtype = x1.dtype
    if "bfloat16" in str(x_dtype) or "float16" in str(x_dtype):
        x1 = x1.astype("float32")
    res = numpy.expm1(x1)
    output_dtypes = bfloat16_conversion(context.param.get("output_dtypes"))
    return res.astype(output_dtypes[0])


def _sign(context: OpInfo):
    x1 = context.param.get("input_arrays")[0]
    x_dtype = x1.dtype
    if "bfloat16" in str(x_dtype) or "float16" in str(x_dtype):
        x1 = x1.astype("float32")
    res = numpy.sign(x1)
    output_dtypes = bfloat16_conversion(context.param.get("output_dtypes"))
    return res.astype(output_dtypes[0])


def _square(context: OpInfo):
    x1 = context.param.get("input_arrays")[0]
    x_dtype = x1.dtype
    if "bfloat16" in str(x_dtype) or "float16" in str(x_dtype):
        x1 = x1.astype("float32")
    res = numpy.square(x1)
    output_dtypes = bfloat16_conversion(context.param.get("output_dtypes"))
    return res.astype(output_dtypes[0])


def _minimum(context: OpInfo):
    x1, x2 = context.param.get("input_arrays")
    x_dtype = x1.dtype
    if "bfloat16" in str(x_dtype) or "float16" in str(x_dtype):
        x1 = x1.astype("float32")
        x2 = x2.astype("float32")
    res = numpy.minimum(x1, x2)
    output_dtypes = bfloat16_conversion(context.param.get("output_dtypes"))
    return res.astype(output_dtypes[0])


def _floor(context: OpInfo):
    x1 = context.param.get("input_arrays")[0]
    x_dtype = x1.dtype
    if "bfloat16" in str(x_dtype) or "float16" in str(x_dtype):
        x1 = x1.astype("float32")
    res = numpy.floor(x1)
    output_dtypes = bfloat16_conversion(context.param.get("output_dtypes"))
    return res.astype(output_dtypes[0])
    

def _cos(context: OpInfo):
    x1 = context.param.get("input_arrays")[0]
    x_dtype = x1.dtype
    if "bfloat16" in str(x_dtype) or "float16" in str(x_dtype):
        x1 = x1.astype("float32")
    res = numpy.cos(x1)
    output_dtypes = bfloat16_conversion(context.param.get("output_dtypes"))
    return res.astype(output_dtypes[0])


def _cosh(context: OpInfo):
    x1 = context.param.get("input_arrays")[0]
    x_dtype = x1.dtype
    if "bfloat16" in str(x_dtype) or "float16" in str(x_dtype):
        x1 = x1.astype("float32")
    res = numpy.cosh(x1)
    output_dtypes = bfloat16_conversion(context.param.get("output_dtypes"))
    return res.astype(output_dtypes[0])


def _sin(context: OpInfo):
    x1 = context.param.get("input_arrays")[0]
    x_dtype = x1.dtype
    if "bfloat16" in str(x_dtype) or "float16" in str(x_dtype):
        x1 = x1.astype("float32")
    res = numpy.sin(x1)
    output_dtypes = bfloat16_conversion(context.param.get("output_dtypes"))
    return res.astype(output_dtypes[0])


def _acos(context: OpInfo):
    x1 = context.param.get("input_arrays")[0]
    x_dtype = x1.dtype
    if "bfloat16" in str(x_dtype) or "float16" in str(x_dtype):
        x1 = x1.astype("float32")
    res = numpy.arccos(x1)
    output_dtypes = bfloat16_conversion(context.param.get("output_dtypes"))
    return res.astype(output_dtypes[0])


def _real(context: OpInfo):
    x = context.param.get("input_arrays")[0]
    if "float16" in context.param.get("stc_input_dtypes"):
        return x
    if "float32" in context.param.get("stc_input_dtypes"):
        return x
    if "complex32" in context.param.get("stc_input_dtypes"):
        xreal, ximag = numpy.split(x, 2, axis=-1)
        return xreal
    if "complex64" in context.param.get("stc_input_dtypes"):
        return x.real
    return None



def _ger(context: OpInfo):
    x1, x2 = context.param.get("input_arrays")[:2]
    res = numpy.outer(x1, x2)
    return res


def _outer(context:OpInfo):
    x1, x2 = context.param.get("input_arrays")[:2]
    res = numpy.outer(x1, x2)
    return res


def _gelu(context: OpInfo):
    input0 = context.param.get("input_arrays")[0]
    input0_dtype = context.param.get("stc_input_dtypes")[0]
    if input0_dtype in ("float16", "bfloat16"):
        input0 = input0.astype("float32")
    input_x = torch.from_numpy(input0)
    out = torch.nn.functional.gelu(input_x, approximate="tanh")
    res = out.numpy()
    output_dtypes = bfloat16_conversion(context.param.get("output_dtypes"))
    return res.astype(output_dtypes[0])


def _gelu_v2(context: OpInfo):
    input_x = context.param.get("input_arrays")[0]
    input_dtype = input_x.dtype
    if input_dtype != "float32":
        input_x = input_x.astype('float32', copy=False)
    erf_input = input_x / numpy.sqrt(2)
    res0 = special.erf(erf_input)
    res1 = (res0 / 2) + 0.5
    res = numpy.multiply(res1, input_x)
    if input_dtype != res.dtype:
        res = res.astype(input_dtype, copy=False)
    return res


def _tanh(context: OpInfo):
    tf.compat.v1.disable_eager_execution()
    input_x = context.param.get("input_arrays")[0]
    input0 = tf.compat.v1.placeholder(shape=input_x.shape,
                                      dtype=input_x.dtype)
    out = tf.math.tanh(input0, name="tanh")
    feed_dict = {input0: input_x}
    init_op = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init_op)
        res = sess.run(out, feed_dict=feed_dict)
    return res


def _tanh_grad(context: OpInfo):
    input0, input1 = context.param.get("input_arrays")
    if input0.dtype == "float16":
        input0 = input0.astype("float32")
        input1 = input1.astype("float32")
    if input0.dtype == tf.bfloat16.as_numpy_dtype:
        input0 = input0.astype("float32")
        input1 = input1.astype("float32")
    data_square = numpy.multiply(input0, input0)
    data_mul = numpy.multiply(data_square, -1)
    data_add = numpy.add(data_mul, 1)
    result = numpy.multiply(data_add, input1)
    if context.param.get("output_dtypes")[0] == "bfloat16":
        return result.astype(tf.bfloat16.as_numpy_dtype, copy=False)
    else:
        return result.astype(context.param.get("output_dtypes")[0], copy=False)


def _sigmoid(context: OpInfo):
    input0 = context.param.get("input_arrays")[0]
    if input0.dtype == tf.bfloat16.as_numpy_dtype:
        input0 = input0.astype("float32")
    tensor_neg = input0 * (-1)
    tensor_exp = numpy.exp(tensor_neg)
    tensor_add = tensor_exp + 1
    res = 1 / tensor_add
    if context.param.get("output_dtypes")[0] == "bfloat16":
        return res.astype(tf.bfloat16.as_numpy_dtype, copy=False)
    else:
        return res.astype(context.param.get("output_dtypes")[0], copy=False)


def _softplus(context: OpInfo):
    input_x = context.param.get("input_arrays")[0]
    dtype_list = [tf.bfloat16.as_numpy_dtype, "float16"]
    if input_x.dtype in dtype_list:
        x = input_x.astype("float32")
    else:
        x = input_x
    input0 = tf.compat.v1.placeholder(shape=x.shape, dtype=x.dtype)
    out = tf.nn.softplus(input0, name="softplus")
    feed_dict = {input0: input_x}
    init_op = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init_op)
        res = sess.run(out, feed_dict=feed_dict)

    if input_x.dtype == tf.bfloat16.as_numpy_dtype:
        return res.astype(tf.bfloat16.as_numpy_dtype)
    else:
        return res.astype(context.param.get("output_dtypes")[0])


def data_processing(data, output_dtype):
    if output_dtype == "float16":
        data = numpy.maximum(data, -65504)
        data = numpy.minimum(data, 65504)
    elif output_dtype == "int8":
        data = numpy.maximum(data, -128)
        data = numpy.minimum(data, 127)
    elif output_dtype == "int32":
        data = numpy.maximum(data, -2147483648)
        data = numpy.minimum(data, 2147483647)
    return data.astype(output_dtype)


def _rsqrt_grad(context: OpInfo):
    input0, input1 = context.param.get("input_arrays")
    output_dtype = context.param.get("output_dtypes")[0]
    if output_dtype in ("int8", "float16", "bfloat16"):
        input0, input1 = input0.astype("float32"), input1.astype("float32")
    rsqrt_const = np.array(-0.5, dtype=input0.dtype)
    res_mul = numpy.multiply(input0, input0)
    res_mul1 = numpy.multiply(res_mul, input0)
    res_mul2 = numpy.multiply(res_mul1, input1)
    res = numpy.multiply(rsqrt_const, res_mul2)
    # res = data_processing(res, output_dtype) #饱和处理
    if output_dtype == "bfloat16":
        res = res.astype(tf.bfloat16.as_numpy_dtype)
    else:
        res = res.astype(output_dtype)
    return res


def _assign_add(context: OpInfo):
    inputs = context.param.get("input_arrays")
    ref = inputs[0]
    value = inputs[1]
    ref_dtype = ref.dtype
    if ref_dtype == tf.bfloat16.as_numpy_dtype:
        ref = ref.astype("float32")
        value = value.astype("float32")
    result = numpy.add(ref, value)

    if context.param.get("output_dtypes")[0] == "bfloat16":
        return result.astype(tf.bfloat16.as_numpy_dtype)
    else:
        return result


def _assign_sub(context: OpInfo):
    var = context.param.get("input_arrays")[0]
    value = context.param.get("input_arrays")[1]
    dtype = value.dtype
    if dtype == tf.bfloat16.as_numpy_dtype:
        var = var.astype("float32")
        value = value.astype("float32")
    result = numpy.subtract(var, value)
    if context.param.get("output_dtypes")[0] == "bfloat16":
        return result.astype(tf.bfloat16.as_numpy_dtype)
    else:
        return result


def _floor_mod(context: OpInfo):
    inputx, inputy = context.param.get("input_arrays")[:2]
    if inputx.dtype == tf.bfloat16.as_numpy_dtype:
        inputx = inputx.astype("float32")
        inputy = inputy.astype("float32")
    x_tensor = torch.from_numpy(inputx)
    y_tensor = torch.from_numpy(inputy)
    res = torch.remainder(x_tensor, y_tensor).numpy()
    if context.param.get("output_dtypes")[0] == "bfloat16":
        return res.astype(tf.bfloat16.as_numpy_dtype)
    return res


def _floor_div(context: OpInfo):
    input0, input1 = context.param.get("input_arrays")[:2]
    output_dtype = context.param.get("output_dtypes")[0]
    if "bfloat16" in str(input0.dtype):
        x, y = input0.astype("float32"), input1.astype("float32")
        res = np.floor(np.divide(x,y))
        return res.astype(input0.dtype, copy=False)
    else:
        res = np.floor(np.divide(input0,input1)).astype(output_dtype)
        return res


def _mod(context: OpInfo):
    input0 = context.param.get("input_arrays")[0]
    input1 = context.param.get("input_arrays")[1]
    return numpy.fmod(input0, input1)


def _add_n(context: OpInfo):
    inputs = context.param.get("input_arrays")
    if context.param.get("output_dtypes")[0] == "bfloat16":
        out = inputs[0].astype("float32")
        for inp in inputs[1:]:
            out = numpy.add(out, inp.astype("float32"))
        out = tf.cast(out, tf.bfloat16)
        with tf.compat.v1.Session() as sess:
            result = sess.run(out)
    else:
        result = inputs[0]
        need_conv = inputs[0].dtype == "float16"
        if need_conv:
            result = inputs[0].astype("float32")
        for inp in inputs[1:]:
            if need_conv:
                inp = inp.astype("float32")
            result = numpy.add(result, inp, dtype=result.dtype)
        if need_conv:
            result = result.astype("float16")
    return result


def _sigmoid_grad(context: OpInfo):
    input0 = context.param.get("input_arrays")[0]
    input1 = context.param.get("input_arrays")[1]
    if input0.dtype == "float16" or input0.dtype == tf.bfloat16.as_numpy_dtype:
        input0 = input0.astype("float32")
        input1 = input1.astype("float32")
    input0 = torch.tensor(input0)
    input1 = torch.tensor(input1)
    res = torch.ops.aten.sigmoid_backward(input1, input0).numpy()

    if context.param.get("output_dtypes")[0] == "bfloat16":
        return res.astype(tf.bfloat16.as_numpy_dtype, copy=False)
    else:
        return res.astype(context.param.get("output_dtypes")[0], copy=False)


def _fill(context: OpInfo):
    input0 = context.param.get("input_arrays")[0]
    dims = context.param.get("dims")
    return numpy.tile(input0, dims)


def _fill_d(context: OpInfo):
    input0 = context.param.get("input_arrays")[0]
    dims = context.param.get("dims")
    res = numpy.tile(input0, dims)
    output_ori_format = context.param.get("output_ori_formats")[0]
    output_format = context.param.get("output_formats")[0]
    output_shape = context.param.get("stc_outputs")[0]
    if transform(res, output_ori_format, output_format, output_shape) is not None:
        res = transform(res, output_ori_format, output_format, output_shape)
    return res


def _tile_d(context: OpInfo):
    x = context.param.get("input_arrays")[0]
    multiples = context.param.get("multiples")
    tensor_x = tf.placeholder(x.dtype, shape=x.shape)
    out = tf.tile(tensor_x, multiples)
    with tf.Session() as sess:
        res = sess.run(out, feed_dict={tensor_x: x})
    output_ori_format = context.param.get("output_ori_formats")[0]
    output_format = context.param.get("output_formats")[0]
    output_shape = context.param.get("stc_outputs")[0]
    if transform(res, output_ori_format, output_format, output_shape) is not None:
        res = transform(res, output_ori_format, output_format, output_shape)
    return res


def _relu(context: OpInfo):
    input0 = context.param.get("input_arrays")[0]
    if "bfloat16" in str(input0.dtype):
        x = input0.astype("float32")
        res = numpy.maximum(x, 0)
        return res.astype(input0.dtype, copy=False)
    else:
        return numpy.maximum(input0, 0)


def _sqrt_grad(context: OpInfo):
    input0, input1 = context.param.get("input_arrays")
    output_dtype = context.param.get("output_dtypes")[0]
    if str(output_dtype)  == "bfloat16":
        input0 = input0.astype("float32")
        input1 = input1.astype("float32")
    div = numpy.divide(input1, input0)
    if output_dtype == "float16":
        div = div.astype("float16")
    res = numpy.multiply(0.5, div)
    if str(output_dtype)  == "bfloat16":
        res = res.astype(tf.bfloat16.as_numpy_dtype)
    return res


def _leaky_relu(context: OpInfo):
    x = context.param.get("input_arrays")[0]
    negative_slope = context.param.get("negative_slope", 0)
    ipt_dtype = context.param.get("stc_input_dtypes")[0]
    out_dtype = context.param.get("output_dtypes")[0]
    if ipt_dtype in ("float16", "bfloat16"):
        x = x.astype("float32")
    x0_r = numpy.where(x >= 0, 1, 0)
    res = x * (numpy.abs(x0_r - 1) * negative_slope + x0_r)

    if out_dtype == "bfloat16":
        return res.astype(tf.bfloat16.as_numpy_dtype, copy=False)
    return res.astype(out_dtype, copy=False)


def _leaky_relu_grad(context: OpInfo):
    x0, x1 = context.param.get("input_arrays")
    out_dtype = context.param.get("output_dtypes")[0]
    negative_slope = context.param.get("negative_slope", 0)
    x0_r = numpy.where(x1 > 0, 1, 0)
    res = x0 * (numpy.abs(x0_r - 1) * negative_slope + x0_r)
    if out_dtype == "bfloat16":
        return res.astype(tf.bfloat16.as_numpy_dtype, copy=False)
    return res.astype(out_dtype, copy=False)


def _power(context: OpInfo):
    input_0 = context.param.get("input_arrays")[0]
    power = context.param.get("power")
    scale = context.param.get("scale")
    shift = context.param.get("shift")
    return (input_0 * scale + shift) ** power


def _fused_mul_add_n(context: OpInfo):
    input0, input1, input2 = context.param.get("input_arrays")
    output_dtype = context.param.get("output_dtypes")[0]
    if str(output_dtype) == "bfloat16":
        input0 = input0.astype("float32")
        input1 = input1.astype("float32")
        input2 = input2.astype("float32")
    input0_holder = tf.compat.v1.placeholder(shape=input0.shape,
                                      dtype=input0.dtype)
    input1_holder = tf.compat.v1.placeholder(shape=input1.shape,
                                      dtype=input1.dtype)
    input2_holder = tf.compat.v1.placeholder(shape=input2.shape,
                                      dtype=input2.dtype)
    output_data1 = tf.multiply(input0_holder, input2_holder)

    output_data = tf.add(output_data1, input1_holder)

    feed_dict = {input0_holder: input0, input1_holder: input1, input2_holder: input2}

    init_op = tf.compat.v1.global_variables_initializer()

    with tf.compat.v1.Session() as sess:
        sess.run(init_op)
        res = sess.run(output_data, feed_dict=feed_dict)
    if str(output_dtype) == "bfloat16":
        res = res.astype(tf.bfloat16.as_numpy_dtype)
    return res 


def _apply_gradient_descent(context: OpInfo):
    var, alpha, delta = context.param.get("input_arrays")
    alpha = numpy.broadcast_to(alpha, var.shape)
    var_change = numpy.multiply(delta, alpha)
    out = numpy.subtract(var, var_change)
    return out


def _cast(context: OpInfo):
    input0 = context.param.get("input_arrays")[0]
    type = context.param.get("stc_input_dtypes")[0]
    dst_type = context.param.get("output_dtypes")[0]
    if type == "float32" and dst_type == "int64":
        out = torch.tensor(input0, dtype=torch.int64).numpy()
        return out
    if dst_type == "complex32":
        _shape = list(input0.shape)
        input0 = input0.reshape(_shape + [1])
        imag = numpy.zeros(_shape + [1], dtype=numpy.float16)
        res = numpy.concatenate((input0, imag), axis=-1)
        return res
    if type == "uint1":
        input0 = np.unpackbits(input0)
    if type == "float32" and dst_type == "float16":
        out = torch.tensor(input0, dtype=torch.float16).numpy()
        return out
    if dst_type == "complex64":
        out = tf.cast(input0, dtype=dst_type)
        with tf.compat.v1.Session() as sess:
            res = sess.run(out)
        return res
    input0_tensor = numpy_to_torch_tensor(input0)
    out_dtype_torch = str_to_dtype(dst_type)
    out = input0_tensor.to(out_dtype_torch)
    if dst_type == "bfloat16":
        out = out.type(torch.float32)
        res = out.numpy().astype(tf.bfloat16.as_numpy_dtype)
    else:
        res = out.numpy()
    return res


def _adds(context: OpInfo):
    input0 = context.param.get("input_arrays")[0]
    value = context.param.get("value")
    if str(input0.dtype) in ("int32", "int64"):
        value = numpy.floor(value).astype(input0.dtype)
    if "bfloat16" in str(input0.dtype):
        x = input0.astype("float32")
        res = numpy.add(x, value)
        return res.astype(tf.bfloat16.as_numpy_dtype,copy=False)
    else:
        res = numpy.add(input0, value)
        return res.astype(context.param.get("output_dtypes")[0], copy=False)


def _div(context: OpInfo):
    input0, input1 = context.param.get("input_arrays")[:2]
    if "bfloat16" in str(input0.dtype):
        x, y = input0.astype("float32"), input1.astype("float32")
        res = x / y
        return res.astype(input0.dtype, copy=False)
    if "complex32" in context.param.get("stc_input_dtypes"):
        x, y = input0, input1
        xreal, ximag = numpy.split(x, 2, axis=-1)
        yreal, yimag = numpy.split(y, 2, axis=-1)
        zreal = (xreal * yreal + ximag * yimag) / (yreal * yreal + yimag * yimag)
        zimag = (ximag * yreal - xreal * yimag) / (yreal * yreal + yimag * yimag)
        res = numpy.concatenate((zreal, zimag), axis=-1)
        return res
    elif "int" in str(input0.dtype):
        res = torch.tensor(input0 / input1)
        res = torch.floor(res)
        res = res.numpy()
        return res
    else:
        return input0 / input1


def _complex(context: OpInfo):
    real, imag = context.param.get("input_arrays")[:2]
    if "float32" in str(real.dtype):
        return real + imag * 1j
    if "float16" in str(real.dtype):
        _shape = list(real.shape) + [1]
        real = real.reshape(_shape)
        imag = imag.reshape(_shape)
        res = numpy.concatenate((real, imag), axis=-1)
        return res
    return None