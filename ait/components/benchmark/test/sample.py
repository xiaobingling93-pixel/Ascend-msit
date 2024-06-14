# Copyright (c) 2023-2023 Huawei Technologies Co., Ltd.
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

import sys
import logging

import aclruntime
import numpy as np

model_path = sys.argv[1]

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# 最短运行样例
def infer_simple():
    device_id = 0
    options = aclruntime.session_options()
    session = aclruntime.InferenceSession(model_path, device_id, options)

    # create new numpy data according inputs info
    barray = bytearray(session.get_inputs()[0].realsize)
    ndata = np.frombuffer(barray)
    # convert numpy to pytensors in device
    tensor = aclruntime.Tensor(ndata)
    tensor.to_device(device_id)

    outnames = [session.get_outputs()[0].name]
    feeds = {session.get_inputs()[0].name: tensor}

    outputs = session.run(outnames, feeds)
    logger.info("outputs:", outputs)

    outarray = []
    for out in outputs:
        # convert acltensor to host memory
        out.to_host()
        # convert acltensor to numpy array
        outarray.append(np.array(out))
    # summary inference throughput
    logger.info("infer avg:{} ms".format(np.mean(session.sumary().exec_time_list)))


# 获取模型信息
def get_model_info():
    device_id = 0
    options = aclruntime.session_options()
    # 方法1设置级别为debug模式后可以打印模型信息
    options.log_level = 1
    session = aclruntime.InferenceSession(model_path, device_id, options)

    # 方法2 直接打印session 也可以获取模型信息
    logger.info(session)

    # 方法3 也可以直接通过get接口去获取
    intensors_desc = session.get_inputs()
    for i, info in enumerate(intensors_desc):
        logger.info(
            "input info i:{} shape:{} type:{} val:{} realsize:{} size:{}".format(
                i, info.shape, info.datatype, int(info.datatype), info.realsize, info.size
            )
        )

    intensors_desc = session.get_outputs()
    for i, info in enumerate(intensors_desc):
        logger.info(
            "outputs info i:{} shape:{} type:{} val:{} realsize:{} size:{}".format(
                i, info.shape, info.datatype, int(info.datatype), info.realsize, info.size
            )
        )


def infer_dynamicshape():
    device_id = 0
    options = aclruntime.session_options()
    session = aclruntime.InferenceSession(model_path, device_id, options)

    # only need call this functon compare infer_simple
    session.set_dynamic_shape("actual_input_1:1,3,224,224")
    session.set_custom_outsize([10000])

    # create new numpy data according inputs info
    barray = bytearray(session.get_inputs()[0].realsize)
    ndata = np.frombuffer(barray)
    # convert numpy to pytensors in device
    tensor = aclruntime.Tensor(ndata)
    tensor.to_device(device_id)

    outnames = [session.get_outputs()[0].name]
    feeds = {session.get_inputs()[0].name: tensor}

    outputs = session.run(outnames, feeds)
    logger.info("outputs:", outputs)

    outarray = []
    for out in outputs:
        # convert acltensor to host memory
        out.to_host()
        # convert acltensor to numpy array
        outarray.append(np.array(out))
    # summary inference throughput
    logger.info("infer avg:{} ms".format(np.mean(session.sumary().exec_time_list)))


# 传入acl文件 执行profiling或者dump
def acljson_run():
    device_id = 0
    options = aclruntime.session_options()
    options.acl_json_path = "./acl.json"
    session = aclruntime.InferenceSession(model_path, device_id, options)


# 并行运行样例
def infer_run_simultaneous():
    device_id = 0

    # one call
    options = aclruntime.session_options()
    session = aclruntime.InferenceSession(model_path, device_id, options)
    # create new numpy data according inputs info
    barray = bytearray(session.get_inputs()[0].realsize)
    ndata = np.frombuffer(barray)
    # convert numpy to pytensors in device
    tensor = aclruntime.Tensor(ndata)
    tensor.to_device(device_id)
    outnames = [session.get_outputs()[0].name]
    feeds = {session.get_inputs()[0].name: tensor}

    # another call
    options1 = aclruntime.session_options()
    model_path1 = sys.argv[2]
    session1 = aclruntime.InferenceSession(model_path1, device_id, options1)
    # create new numpy data according inputs info
    barray1 = bytearray(session1.get_inputs()[0].realsize)
    ndata1 = np.frombuffer(barray1)
    # convert numpy to pytensors in device
    tensor1 = aclruntime.Tensor(ndata1)
    tensor1.to_device(device_id)
    outnames1 = [session1.get_outputs()[0].name]
    feeds1 = {session1.get_inputs()[0].name: tensor1}

    # one run
    outputs = session.run(outnames, feeds)
    logger.info("outputs:", outputs)

    outarray = []
    for out in outputs:
        # convert acltensor to host memory
        out.to_host()
        # convert acltensor to numpy array
        outarray.append(np.array(out))
    # summary inference throughput
    logger.info("infer avg:{} ms".format(np.mean(session.sumary().exec_time_list)))

    # another run
    outputs1 = session1.run(outnames1, feeds1)
    logger.info("outputs1:", outputs1)

    outarray1 = []
    for out in outputs1:
        # convert acltensor to host memory
        out.to_host()
        # convert acltensor to numpy array
        outarray1.append(np.array(out))
    # summary inference throughput
    logger.info("infer avg:{} ms".format(np.mean(session1.sumary().exec_time_list)))


def infer_dynamic_dims():
    device_id = 0
    options = aclruntime.session_options()
    session = aclruntime.InferenceSession(model_path, device_id, options)

    # only need call this functon compare infer_simple
    session.set_dynamic_dims("actual_input_1:1,3,224,224")

    # create new numpy data according inputs info
    barray = bytearray(session.get_inputs()[0].realsize)
    ndata = np.frombuffer(barray)
    # convert numpy to pytensors in device
    tensor = aclruntime.Tensor(ndata)
    tensor.to_device(device_id)

    outnames = [session.get_outputs()[0].name]
    feeds = {session.get_inputs()[0].name: tensor}

    outputs = session.run(outnames, feeds)
    logger.info("outputs:", outputs)

    outarray = []
    for out in outputs:
        # convert acltensor to host memory
        out.to_host()
        # convert acltensor to numpy array
        outarray.append(np.array(out))
    # summary inference throughput
    logger.info("infer avg:{} ms".format(np.mean(session.sumary().exec_time_list)))


def infer_dynamics_hw():
    device_id = 0
    options = aclruntime.session_options()
    session = aclruntime.InferenceSession(model_path, device_id, options)

    # only need call this functon compare infer_simple
    session.set_dynamic_hw(224, 224)

    # create new numpy data according inputs info
    barray = bytearray(session.get_inputs()[0].realsize)
    ndata = np.frombuffer(barray)
    # convert numpy to pytensors in device
    tensor = aclruntime.Tensor(ndata)
    tensor.to_device(device_id)

    outnames = [session.get_outputs()[0].name]
    feeds = {session.get_inputs()[0].name: tensor}

    outputs = session.run(outnames, feeds)
    logger.info("outputs:", outputs)

    outarray = []
    for out in outputs:
        # convert acltensor to host memory
        out.to_host()
        # convert acltensor to numpy array
        outarray.append(np.array(out))
    # summary inference throughput
    logger.info("infer avg:{} ms".format(np.mean(session.sumary().exec_time_list)))


def infer_dynamic_batchsize():
    device_id = 0
    options = aclruntime.session_options()
    session = aclruntime.InferenceSession(model_path, device_id, options)

    # only need call this functon compare infer_simple
    # dynamic_batchsize can be 1, 2, 4, 8
    session.set_dynamic_batchsize(2)

    # create new numpy data according inputs info
    barray = bytearray(session.get_inputs()[0].realsize)
    ndata = np.frombuffer(barray)
    # convert numpy to pytensors in device
    tensor = aclruntime.Tensor(ndata)
    tensor.to_device(device_id)

    outnames = [session.get_outputs()[0].name]
    feeds = {session.get_inputs()[0].name: tensor}

    outputs = session.run(outnames, feeds)
    logger.info("outputs:", outputs)

    outarray = []
    for out in outputs:
        # convert acltensor to host memory
        out.to_host()
        # convert acltensor to numpy array
        outarray.append(np.array(out))
    # summary inference throughput
    logger.info("infer avg:{} ms".format(np.mean(session.sumary().exec_time_list)))


infer_simple()
