# -*- coding: utf-8 -*-
# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
import numpy as np
import pytest

from msserviceprofiler.modelevalstate.config.config import OptimizerConfigField, PerformanceIndex
from msserviceprofiler.modelevalstate.optimizer.experience_fine_tunning import MindIeFineTune, StopFineTune


def test_update_request_rate():
    mindie_fine_tune = MindIeFineTune(ttft_penalty=1, tpot_penalty=1)
    my_support_field = [
        OptimizerConfigField(name="CONCURRENCY", config_position="env", min=10, max=1001, dtype="int", value=200),
        OptimizerConfigField(name="REQUESTRATE", config_position="env", min=0, max=1001, dtype="int", value=20),
    ]
    proportion = 0.05
    _flag = mindie_fine_tune.update_request_rate(my_support_field, proportion)
    assert _flag
    assert my_support_field[-1].value == 21
    proportion = -0.05
    _flag = mindie_fine_tune.update_request_rate(my_support_field, proportion)
    assert _flag
    assert my_support_field[-1].value == 19.95



def test_ttft_gt_tp_gt_ttft_diff_gt_tp_diff():
    """测试ttft >,tpot > 并且ttft差值大，调小prefill size的情况"""
    my_support_field = [
        OptimizerConfigField(name="CONCURRENCY", config_position="env", min=10, max=1001, dtype="int", value=200),
        OptimizerConfigField(name="REQUESTRATE", config_position="env", min=0, max=1001, dtype="int", value=20),
    ]
    params = np.array([100, 20])
    mindie_fine_tune = MindIeFineTune(ttft_penalty=1, tpot_penalty=1, target_field=tuple(my_support_field))
    performance_index = PerformanceIndex()
    performance_index.time_to_first_token = 0.5
    performance_index.time_per_output_token = 0.05
    # 满足slo
    with pytest.raises(StopFineTune):
        mindie_fine_tune.mindie_fine_tune_with_cycle(params, performance_index)
    performance_index.time_to_first_token = 0.55
    performance_index.time_per_output_token = 0.055
    with pytest.raises(StopFineTune):
        mindie_fine_tune.mindie_fine_tune_with_cycle(params, performance_index)
    mindie_fine_tune.ttft_penalty = 0
    mindie_fine_tune.tpot_penalty = 0
    with pytest.raises(StopFineTune):
        mindie_fine_tune.mindie_fine_tune_with_cycle(params, performance_index)
    # 测试只限制tpot场景
    mindie_fine_tune.tpot_penalty = 3.0
    with pytest.raises(StopFineTune):
        mindie_fine_tune.mindie_fine_tune_with_cycle(params, performance_index)
    performance_index.time_per_output_token = 0.06
    result = mindie_fine_tune.mindie_fine_tune_with_cycle(params, performance_index)
    assert result[-1].value < my_support_field[-1].value
    performance_index.time_per_output_token = 0.04
    result = mindie_fine_tune.mindie_fine_tune_with_cycle(params, performance_index)
    assert result[-1].value > my_support_field[-1].value
    # 测试限制tpot和ttft
    mindie_fine_tune.ttft_penalty = 3.0
    performance_index.time_per_output_token = 0.06
    performance_index.time_to_first_token = 0.58
    result = mindie_fine_tune.mindie_fine_tune_with_cycle(params, performance_index)
    assert result[-1].value == my_support_field[-1].value * (1 - 0.01 / 0.05 * 0.5)
    performance_index.time_per_output_token = 0.058
    performance_index.time_to_first_token = 0.6
    result = mindie_fine_tune.mindie_fine_tune_with_cycle(params, performance_index)
    assert result[-1].value == my_support_field[-1].value * (1 - 0.1 / 0.5 * 0.5)
    performance_index.time_per_output_token = 0.06
    performance_index.time_to_first_token = 0.4
    result = mindie_fine_tune.mindie_fine_tune_with_cycle(params, performance_index)
    assert result[-1].value == my_support_field[-1].value * (1 - 0.01 / 0.05 * 0.5)
    performance_index.time_per_output_token = 0.04
    performance_index.time_to_first_token = 0.6
    result = mindie_fine_tune.mindie_fine_tune_with_cycle(params, performance_index)
    assert result[-1].value == my_support_field[-1].value * (1 - 0.1 / 0.5 * 0.5)
    performance_index.time_per_output_token = 0.04
    performance_index.time_to_first_token = 0.3
    result = mindie_fine_tune.mindie_fine_tune_with_cycle(params, performance_index)
    assert result[-1].value == my_support_field[-1].value * (1 + 0.01 / 0.05 * 0.5)
    performance_index.time_per_output_token = 0.03
    performance_index.time_to_first_token = 0.4
    result = mindie_fine_tune.mindie_fine_tune_with_cycle(params, performance_index)
    assert result[-1].value == my_support_field[-1].value * (1 + 0.1 / 0.5 * 0.5)