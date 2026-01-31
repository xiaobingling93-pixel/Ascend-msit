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
import logging
import os

logger = logging.getLogger(__name__)
LOG_LEVEL = os.environ.get("LOG_LEVEL", 'ERROR')

logging.basicConfig(
    level=LOG_LEVEL,
    format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %I:%M:%S %p'
)

try:
    import torch
    import torchvision
except Exception as e:
    logger.error("Something wrong with `torch` or `torchvision`. Try reinstall if their versions are not compatible.")
    raise


class LargeNet(torch.nn.Module):
    def __init__(self):
        super(LargeNet, self).__init__()
        self.base_model = torchvision.models.resnet152()
        self.fc1 = torch.nn.Linear(1000, 2048)
        self.fc2 = torch.nn.Linear(2048, 4096)
        self.fc3 = torch.nn.Linear(4096, 4096 * 2)
        self.fc4 = torch.nn.Linear(4096 * 2, 4096 * 4)
        self.fc5 = torch.nn.Linear(4096 * 4, 4096 * 10)

    def forward(self, x):
        x = self.base_model(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))

        return x
    

if __name__ == "__main__":
    try:
        import onnx
    except Exception as e:
        logger.error("onnx error. Try reinstall onnx.")
        raise

    try:
        os.mkdir("temp_dir", 0o750)
    except Exception as e:
        logger.error("%s cannot be created. See if it exists or do not have permission.", "temp_dir")
        raise

    logger.debug("%s has being created successfully.", os.path.abspath("temp_dir"))

    model = LargeNet()

    try:
        torch.onnx.export(model, torch.randn(1, 3, 2, 4), "temp_dir/large_model.onnx")
    except Exception as e:
        logger.error("Error occured when trying to save the onnx model.")
    else:
        logger.debug("%s has being saved successfully", os.path.abspath("temp_dir/large_model.onnx"))
