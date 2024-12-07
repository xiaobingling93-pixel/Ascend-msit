# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.

import torch
import torch.nn as nn


def conv_bn_relu(in_channel, out_channel, kernel_size, stride):
    return  nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size, stride),
        nn.BatchNorm2d(out_channel),
        nn.Relu()
    )


class TestAscendQuantModel(nn.Module):









    