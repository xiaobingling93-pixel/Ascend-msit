# -*- coding: utf-8 -*-
# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
import math
import numpy as np
import pytest

from msserviceprofiler.modelevalstate.config.config import OptimizerConfigField, PerformanceIndex
from msserviceprofiler.modelevalstate.optimizer.experience_fine_tunning import FineTune, StopFineTune


def test_update_field():
    fine_tune = FineTune(ttft_penalty=1, tpot_penalty=1)
    my_support_field = [
        OptimizerConfigField(name="CONCURRENCY", config_position="env", min=10, max=1001, dtype="int", value=200),
        OptimizerConfigField(name="REQUESTRATE", config_position="env", min=0, max=1001, dtype="int", value=20),
    ]
    proportion = 0.05
    _flag = fine_tune.update_field(my_support_field, proportion)
    assert _flag
    assert my_support_field[-1].value == 21
    proportion = -0.05
    _flag = fine_tune.update_field(my_support_field, proportion)
    assert _flag
    assert math.isclose(my_support_field[-1].value, 19.95, rel_tol=1e-9)



def test_ttft_gt_tp_gt_ttft_diff_gt_tp_diff():
    """测试ttft >,tpot > 并且ttft差值大，调小prefill size的情况"""
    my_support_field = [
        OptimizerConfigField(name="CONCURRENCY", config_position="env", min=10, max=1001, dtype="int", value=200),
        OptimizerConfigField(name="REQUESTRATE", config_position="env", min=0, max=1001, dtype="int", value=20),
    ]
    params = np.array([100, 20])
    mindie_fine_tune = FineTune(ttft_penalty=1, tpot_penalty=1, target_field=tuple(my_support_field))
    performance_index = PerformanceIndex()
    performance_index.time_to_first_token = 0.5
    performance_index.time_per_output_token = 0.05
    # 满足slo
    with pytest.raises(StopFineTune):
        mindie_fine_tune.fine_tune_with_concurrency(params, performance_index)
    performance_index.time_to_first_token = 0.55
    performance_index.time_per_output_token = 0.055
    with pytest.raises(StopFineTune):
        mindie_fine_tune.fine_tune_with_concurrency(params, performance_index)
    mindie_fine_tune.ttft_penalty = 0
    mindie_fine_tune.tpot_penalty = 0
    with pytest.raises(StopFineTune):
        mindie_fine_tune.fine_tune_with_concurrency(params, performance_index)
    # 测试只限制tpot场景
    mindie_fine_tune.tpot_penalty = 3.0
    with pytest.raises(StopFineTune):
        mindie_fine_tune.fine_tune_with_concurrency(params, performance_index)
    performance_index.time_per_output_token = 0.06
    result = mindie_fine_tune.fine_tune_with_concurrency(params, performance_index)
    assert result[-1].value < my_support_field[-1].value
    performance_index.time_per_output_token = 0.04
    result = mindie_fine_tune.fine_tune_with_concurrency(params, performance_index)
    assert result[-1].value < my_support_field[-1].value
    # 测试限制tpot和ttft
    mindie_fine_tune.ttft_penalty = 3.0
    performance_index.time_per_output_token = 0.06
    performance_index.time_to_first_token = 0.58
    result = mindie_fine_tune.fine_tune_with_concurrency(params, performance_index)
    assert result[-1].value == 0
    performance_index.time_per_output_token = 0.058
    performance_index.time_to_first_token = 0.6
    result = mindie_fine_tune.fine_tune_with_concurrency(params, performance_index)
    assert result[-1].value == 0
    performance_index.time_per_output_token = 0.06
    performance_index.time_to_first_token = 0.4
    result = mindie_fine_tune.fine_tune_with_concurrency(params, performance_index)
    assert result[-1].value == 0


def test_fine_tune_with_cycle_update_concurrency():
    """测试ttft >,tpot > 并且ttft差值大，调小prefill size的情况"""
    my_support_field = [
        OptimizerConfigField(name="CONCURRENCY", config_position="env", min=10, max=1001, dtype="int", value=200),
        OptimizerConfigField(name="REQUESTRATE", config_position="env", min=10, max=20, dtype="int", value=10),
    ]
    params = np.array([200, 10])
    fine_tune = FineTune(target_field=tuple(my_support_field), ttft_penalty=3, tpot_penalty=3)
    performance_index = PerformanceIndex()
    performance_index.time_to_first_token = 0.7
    performance_index.time_per_output_token = 0.05
    result = fine_tune.fine_tune_with_concurrency(params, performance_index)
    assert result[0].value == my_support_field[0].value * 0.5
 
 
def test_fine_tune_with_concurrency():
    my_support_field = [
        OptimizerConfigField(name="max_batch_size", config_position="env", min=100, max=800, dtype="int"),
        OptimizerConfigField(name="CONCURRENCY", config_position="env", min=10, max=1001, dtype="int", value=200),
        OptimizerConfigField(name="REQUESTRATE", config_position="env", min=0, max=1001, dtype="int", value=20),
    ]
    fine_tune = FineTune(target_field=tuple(my_support_field), ttft_penalty=3, tpot_penalty=3)
 
    # 测试tpot和ttft都超过upper_bound的情况
    params = np.array([100, 200, 20])
    performance_index = PerformanceIndex(time_per_output_token=100, time_to_first_token=100)
    result = fine_tune.fine_tune_with_concurrency(params, performance_index)
    assert result[-1].value == 0
    assert result[-2].value == 100
 
    # 测试tpot和ttft都低于lower_bound的情况
    params = np.array([100, 200, 20])
    performance_index = PerformanceIndex(time_per_output_token=0.03, time_to_first_token=0.3)
    fine_tune.fine_tune_with_concurrency(params, performance_index)
    result = fine_tune.fine_tune_with_concurrency(params, performance_index)
    assert result[-1].value == 0
    assert result[-2].value == 400
 
    # 测试tpot和ttft都在lower_bound和upper_bound之间的情况
    params = np.array([100, 200, 20])
    performance_index = PerformanceIndex(time_per_output_token=0.05, time_to_first_token=0.5)
    with pytest.raises(StopFineTune):
        result = fine_tune.fine_tune_with_concurrency(params, performance_index)