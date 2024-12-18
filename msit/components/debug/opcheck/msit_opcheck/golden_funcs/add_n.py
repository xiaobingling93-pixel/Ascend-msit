# Copyright (c) 2024-2025 Huawei Technologies Co., Ltd.
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

from msit_opcheck.graph_parser import OpInfo


def _add_n(context: OpInfo):
    inputs = context.param.get("input_arrays")
    if context.param.get("output_dtypes")[0] == "bfloat16":
        import tensorflow as tf
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