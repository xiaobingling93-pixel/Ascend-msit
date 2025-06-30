# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
from unittest.mock import MagicMock, patch

import pytest
import msserviceprofiler
sys.path.insert(0, os.path.join(msserviceprofiler.__path__[0], "vllm_profiler"))  # skip importing from __init__
from vllm_profiler_core.model_hookers import (
    ExecutorBaseExecuteModelHook,
    ModelRunnerExecuteHook,
    ModelForwardHook,
    SetForwardContextHook,
    GLOBAL_FORWARD_PROF,
)


def test_executor_base_init():
    fake_executor_base = MagicMock()
    fake_executor_base.execute_model = MagicMock()
    fake_dist_executor_base = MagicMock()
    fake_dist_executor_base.execute_model = MagicMock()

    vllm_executor_executor_base = MagicMock(
        ExecutorBase=fake_executor_base,
        DistributedExecutorBase=fake_dist_executor_base
    )

    sys.modules["vllm.executor.executor_base"] = vllm_executor_executor_base

    hooker = ExecutorBaseExecuteModelHook()

    with patch.object(hooker, "do_hook") as mock_do_hook:
        hooker.init()

        assert mock_do_hook.call_count == 2
        # Check first call for ExecutorBase
        hook_points1, profiler_func_maker1 = mock_do_hook.call_args_list[0][0]
        assert hook_points1 == [fake_executor_base.execute_model]
        assert callable(profiler_func_maker1)
        
        # Check second call for DistributedExecutorBase
        hook_points2, profiler_func_maker2 = mock_do_hook.call_args_list[1][0]
        assert hook_points2 == [fake_dist_executor_base.execute_model]
        assert callable(profiler_func_maker2)


def test_executor_base_execute_model_maker():
    mock_ori_func = MagicMock(return_value="Result")

    mock_this = MagicMock()
    mock_execute_model_req = MagicMock()
    
    # Setup mock sequence group metadata
    mock_seq_metadata1 = MagicMock()
    mock_seq_metadata1.request_id = "req1"
    mock_seq_metadata1.is_prompt = True
    mock_seq_data1 = MagicMock()
    mock_seq_data1.get_len.return_value = 10
    mock_seq_data1.prompt_token_ids = [1, 2, 3]
    mock_seq_metadata1.seq_data = {"seq1": mock_seq_data1}
    
    mock_seq_metadata2 = MagicMock()
    mock_seq_metadata2.request_id = "req2"
    mock_seq_metadata2.is_prompt = False
    mock_seq_data2 = MagicMock()
    mock_seq_data2.get_len.return_value = 5
    mock_seq_data2.prompt_token_ids = [1, 2]
    mock_seq_metadata2.seq_data = {"seq2": mock_seq_data2}
    
    mock_execute_model_req.seq_group_metadata_list = [mock_seq_metadata1, mock_seq_metadata2]

    hooker = ExecutorBaseExecuteModelHook()
    with patch.object(hooker, "do_hook") as mock_do_hook:
        hooker.init()

        mock_do_hook.assert_called()
        _, execute_model_maker = mock_do_hook.call_args_list[0][0]
        execute_model = execute_model_maker(mock_ori_func)

        # Mock Profiler methods
        with patch('ms_service_profiler.Profiler') as mock_profiler:
            mock_profiler_instance = MagicMock()
            mock_profiler.return_value = mock_profiler_instance
            
            result = execute_model(mock_this, mock_execute_model_req)

            # Verify original function was called
            mock_ori_func.assert_called_once()
            assert result == "Result"


def test_execute_init():
    fake_model_runner = MagicMock()
    fake_model_runner.execute_model = MagicMock()

    vllm_work_model_runner = MagicMock(ModelRunner=fake_model_runner)

    sys.modules["vllm.worker.model_runner"] = vllm_work_model_runner

    hooker = ModelRunnerExecuteHook()

    with patch.object(hooker, "do_hook") as mock_do_hook:
        hooker.init()

        mock_do_hook.assert_called_once()
        hook_points, profiler_func_maker = mock_do_hook.call_args[0]
        assert hook_points == [fake_model_runner.execute_model]
        assert callable(profiler_func_maker)


def test_execute_model_maker():
    mock_ori_func = MagicMock(side_effect=["Result", "SecondResult"])

    mock_this = MagicMock()
    mock_model_input = MagicMock()
    mock_model_input.atten_metadata.prefill_metadata = True
    mock_model_input.request_ids_to_seq_ids.item.return_value = [("req1", "seq1")]
    mock_model_input.input_tokens.shape = [5, 0]

    hooker = ModelRunnerExecuteHook()
    with patch.object(hooker, "do_hook") as mock_do_hook:
        hooker.init()

        mock_do_hook.assert_called_once()
        _, execute_model_maker = mock_do_hook.call_args[0]
        execute_model = execute_model_maker(mock_ori_func)

        # 第一次执行，不进入profiler逻辑
        assert hooker.is_model_first_run is True
        result = execute_model(mock_this, mock_model_input, MagicMock())
        mock_ori_func.assert_called_once()
        assert result == "Result"

        # 第二次执行，进入profiler逻辑
        result = execute_model(mock_this, mock_model_input, MagicMock())
        assert result == "SecondResult"


def test_forward_init():
    fake_common_attention_state = MagicMock()
    fake_common_attention_state.begin_forward = MagicMock()

    vllm_attention_backends_utils = MagicMock(CommonAttentionState=fake_common_attention_state)

    sys.modules["vllm.attention.backends.utils"] = vllm_attention_backends_utils

    hooker = ModelForwardHook()

    with patch.object(hooker, "do_hook") as mock_do_hook:
        hooker.init()

        mock_do_hook.assert_called_once()
        hook_points, profiler_func_maker = mock_do_hook.call_args[0]
        assert hook_points == [fake_common_attention_state.begin_forward]
        assert callable(profiler_func_maker)


def test_begin_forward_maker():
    mock_ori_func = MagicMock(side_effect=["Result", "SecondResult"])

    mock_this = MagicMock()
    mock_model_input = MagicMock()
    mock_model_input.request_ids_to_seq_ids.item.return_value = [("req1", "seq1")]

    hooker = ModelForwardHook()
    with patch.object(hooker, "do_hook") as mock_do_hook:
        hooker.init()

        mock_do_hook.assert_called_once()
        _, begin_forward_maker = mock_do_hook.call_args[0]
        begin_forward = begin_forward_maker(mock_ori_func)

        # 第一次执行，不进入profiler逻辑
        assert hooker.is_forward_first_run is True
        result = begin_forward(mock_this, mock_model_input)
        mock_ori_func.assert_called_once()
        assert result == "Result"

        # 第二次执行，进入profiler逻辑
        result = begin_forward(mock_this, mock_model_input)
        assert result == "SecondResult"


def test_set_forward_context_init():
    sys.modules["vllm"] = MagicMock(forward_context=MagicMock())
    
    hooker = SetForwardContextHook()
    with patch.object(hooker, "do_hook") as mock_do_hook:
        hooker.init()
        
        mock_do_hook.assert_called_once()
        hook_point, context_maker = mock_do_hook.call_args[0]
        assert hook_point == [sys.modules["vllm"].forward_context.set_forward_context]
        assert callable(context_maker)


def test_set_forward_context_maker():
    # 准备全局变量
    global GLOBAL_FORWARD_PROF
    GLOBAL_FORWARD_PROF = [MagicMock()]
    
    mock_ori_context = MagicMock()
    mock_ori_context.return_value.__enter__.return_value = None
    mock_ori_context.return_value.__exit__.return_value = None
    
    hooker = SetForwardContextHook()
    with patch.object(hooker, "do_hook") as mock_do_hook:
        hooker.init()
        context_maker = mock_do_hook.call_args[0][1](mock_ori_context)
        
        # 调用context manager
        with context_maker() as ctx:
            # 验证原始context被调用
            mock_ori_context.assert_called_once()
