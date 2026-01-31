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

from auto_optimizer.inference_engine.post_process.post_process_base import PostProcessBase
from auto_optimizer.inference_engine.data_process_factory import PostProcessFactory
from components.debug.common import logger


@PostProcessFactory.register("classification")
class ClassificationPostProcess(PostProcessBase, ABC):

    def __call__(self, loop, cfg, in_queue, out_queue):
        """
        和基类的参数顺序和个数需要一致
        """
        logger.debug("post_process start")
        try:
            for _ in range(loop):
                data = in_queue.get()

                out_queue.put(data)
        except Exception as err:
            logger.error("post_process failed error={}".format(err))

        logger.debug("post_process end")
