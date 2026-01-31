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
from abc import abstractmethod


class PostProcessBase(object):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, loop, cfg, in_queue, out_queue):
        """
        loop: 循环次数，根据数据集大小、batch_size及worker计算得到loop次数
        cfg: 配置文件，参考auto_optimizer\configs\cv\classification\example.py
        in_queue: 输入数据队列
        out_queue： 输出数据队列
        数据队列建议存放数据格式：[[batch_lable], [[batch_data_0], [batch_data_1]]]
        batch_lable：表示多batch时，对应数据集的label，用于精度评测
        batch_data_n：表示第n个输入or输出，batch_data_n包含batch组数据
        """
        pass
