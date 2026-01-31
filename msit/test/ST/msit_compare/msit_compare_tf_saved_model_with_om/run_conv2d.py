# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------
import logging
import os
import tensorflow as tf

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

# 获取 logger 实例
logger = logging.getLogger("saved_model_logger")


def convert_saved_model_to_pb(saved_model_path, pb_model_path):
    sess = tf.compat.v1.keras.backend.get_session()
    model = tf.compat.v1.saved_model.load(sess, {'serve'}, saved_model_path)
    signature_outputs = model.signature_def.get('serving_default').outputs
    output_name = ""
    for tensor_info in signature_outputs.values():
        output_name = tensor_info.name.split(':')[0]

    frozen_graph = tf.compat.v1.graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(),
                                                                          [output_name])
    frozen_graph = tf.compat.v1.graph_util.remove_training_nodes(frozen_graph)
    try:
        with open(pb_model_path, "wb") as ofile:
            ofile.write(frozen_graph.SerializeToString())
        logger.info("saved_model convert pb success")
    except Exception as e:
        logger.error("saved_model convert pb false")


if __name__ == '__main__':
    model_path = f"{os.environ['PROJECT_PATH']}/resource/msit_compare/saved_model/model/conv2D"
    pb_path = f"{os.environ['PROJECT_PATH']}/resource/msit_compare/saved_model/model/conv2D.pb"
    # saved_model to pb
    convert_saved_model_to_pb(model_path, pb_path)