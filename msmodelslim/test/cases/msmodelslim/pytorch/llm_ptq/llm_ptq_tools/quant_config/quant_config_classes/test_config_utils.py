# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

import pytest

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_config.quant_config_classes.config_utils import \
    check_and_generate_config_param
from ..old_quant_config import OldQuantConfig


class TestConfigUtils:
    def test_check_and_generate_config_param_when_default_value(self):
        old_quant_config = OldQuantConfig()
        check_and_generate_config_param(old_quant_config)
        assert True

    def test_check_and_generate_config_param_should_raise_error_when_error_wbit_value(self):
        old_quant_config = OldQuantConfig()
        old_quant_config.w_bit = 1
        with pytest.raises(ValueError):
            check_and_generate_config_param(old_quant_config)
