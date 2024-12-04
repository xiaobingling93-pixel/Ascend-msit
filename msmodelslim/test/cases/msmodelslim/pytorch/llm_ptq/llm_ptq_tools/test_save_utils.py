# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

import os
import shutil
import torch
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import save_utils


save_utils.ONE_GB_FILE_BYTES = 1


class TestSaveFilePartial:
    def test_func(self):
        weight_dict = {
            'w0': torch.randn(1, 4),  # 1*4*4 = 16
            'w1': torch.randn(1, 4),
            'w2': torch.randn(1, 4),
            'w3': torch.randn(1, 4),
            'w4': torch.randn(1, 4),
            'w5': torch.randn(1, 4),
        }

        TEST_SAVE_PATH = 'tmp_file.safetensors'

        save_utils.save_file_partial(weight_dict, TEST_SAVE_PATH, 32)

        if os.path.exists(TEST_SAVE_PATH):
            shutil.rmtree(TEST_SAVE_PATH)
