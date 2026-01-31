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
from abc import ABC

import numpy as np
from tqdm import tqdm

from auto_optimizer.inference_engine.evaluate.evaluate_base import EvaluateBase
from auto_optimizer.inference_engine.data_process_factory import EvaluateFactory
from components.debug.common import logger


@EvaluateFactory.register("classification")
class ClassificationEvaluate(EvaluateBase, ABC):

    def __call__(self, loop, batch_size, cfg, in_queue):
        """
        和基类的参数顺序和个数需要一致
        """
        logger.debug("evaluate start")
        try:
            topk = ClassificationEvaluate._get_params(cfg)
        except Exception as err:
            logger.error("evaluate failed error={}".format(err))
            raise RuntimeError("evaluate failed error") from err
        count = 0
        count_hit = np.zeros([len(topk), loop * batch_size])
        for _ in tqdm(range(loop), desc="Evaluating"):
            in_data = in_queue.get()
            if len(in_data) < 2:  # include lable and data
                raise RuntimeError("input params error len={}".format(len(in_data)))
            labels, data = in_data[0], in_data[1]

            try:
                # 多batch下，按batch取出对应的数据，先判断文件名称和模型输出数据大小是否一致
                if len(data[0]) != len(labels):
                    raise RuntimeError("input params error len={}".format(len(data[0])))
            except IndexError as e:
                raise RuntimeError(f"index error occurred: {str(e)}") from e

            for index, label in enumerate(labels):
                for idx, k in enumerate(topk):
                    hit = self._is_hit_ground_truth([data[0][index]], label, k)

                    if hit:
                        count_hit[idx][count] = 1
                count += 1

        for idx, k in enumerate(topk):
            if count != 0:
                top_k_accuracy = np.count_nonzero(count_hit[idx]) / count
                logger.info("top_{}_accuracy={}".format(k, top_k_accuracy))

        logger.debug("evaluate end")

    @staticmethod
    def _get_params(cfg):
        try:
            topk = cfg["topk"]
            return topk
        except Exception as err:
            raise RuntimeError("get params failed error={}".format(err)) from err

    def _is_hit_ground_truth(self, data, targets, k):
        values, indices = self._get_top_k(data, k)
        for indice in indices:
            if indice == int(targets):
                return True

        return False

    def _get_top_k(self, data, k, order=True):
        indices = np.argsort(data, axis=-1)[:, -k:]
        if order:
            temp = []
            for indice in indices:
                temp.append(indice[::-1])
            indices = np.array(temp)
        values = []
        for idx, item in zip(indices, data):
            value = item.reshape(1, -1)[:, idx].reshape(-1)
            values.append(value)
        return np.array(values)[0], indices[0]
