# -*- coding: utf-8 -*-
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
import torch
import torchvision.models as models
import mindietorch

from msit_llm import DumpConfig, register_hook

# 创建随机输入数据，尺寸为（1, 3, 224, 224）对应于（batch_size, channels, height, width）
input_data = torch.randn(1, 3, 224, 224)

# 加载 ResNet-50 模型
model = models.resnet50(pretrained=True)
model.eval()  # 设置模型为评估模式

dump_config = DumpConfig(dump_path="./dump/cpu")
register_hook(model, dump_config)  # model是要dump中间tensor的模型实例，在模型初始化后添加代码
result_cpu = model.forward(input_data)

# compile 模型
input_info = [mindietorch.Input((1, 3, 224, 224))]
traced_model = torch.jit.trace(model, input_data)
compiled_module = mindietorch.compile(traced_model, inputs=input_info, soc_version="<soc_version>")  # 替换成用户的具体型号

mindietorch.set_device(0)
result = compiled_module(input_data.to("npu:0")).cpu()
mindietorch.finalize()
