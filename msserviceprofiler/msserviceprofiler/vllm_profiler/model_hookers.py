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
from contextlib import contextmanager
from ms_service_profiler import Profiler, Level
from .module_hook import vllm_hook, recover_hooks_for
import threading

# 线程安全的全局状态
class HookState:
    __slots__ = ('forward_profiler', 'execute_model_first_run', 'begin_forward_first_run')

    def __init__(self):
        self.forward_profiler = None
        self.execute_model_first_run = True
        self.begin_forward_first_run = True

# 线程本地存储
_thread_local = threading.local()

def _get_state() -> HookState:
    """获取线程本地状态"""
    if not hasattr(_thread_local, 'hook_state'):
        _thread_local.hook_state = HookState()
    return _thread_local.hook_state

@vllm_hook(
    hook_points=[
        "vllm.executor.executor_base.ExecutorBase.execute_model",
        "vllm.executor.executor_base.DistributedExecutorBase.execute_model"
    ],
    min_version="0.8.4"
)
def handle_execute_model(original_func, this, execute_model_req, *args, **kwargs):
    """处理执行模型钩子"""
    state = _get_state()

    # 跳过首次执行（初始化阶段）
    if state.execute_model_first_run:
        state.execute_model_first_run = False
        return original_func(this, execute_model_req, *args, **kwargs)

    # 准备性能分析
    prof = Profiler(Level.INFO).domain("ModelExecute")
    is_prefill = False
    request_data = []

    for seq_metadata in execute_model_req.seq_group_metadata_list:
        if seq_metadata.seq_data:
            seq = next(iter(seq_metadata.seq_data.values()))
            iter_size = seq.get_len() - len(seq.prompt_token_ids)
        else:
            iter_size = 0

        request_data.append({
            "rid": seq_metadata.request_id,
            "iter_size": iter_size
        })

        is_prefill = is_prefill or seq_metadata.is_prompt

    # 设置分析器属性
    prof.res(request_data)
    prof.attr("batch_type", "Prefill" if is_prefill else "Decode")
    prof.attr("batch_size", len(execute_model_req.seq_group_metadata_list))

    # 记录预处理事件
    Profiler(Level.INFO).domain("ModelExecute").res(
        [{"rid": md.request_id} for md in execute_model_req.seq_group_metadata_list]
    ).event("preprocess")

    # 执行原始函数并记录耗时
    with prof.span("modelExec"):
        return original_func(this, execute_model_req, *args, **kwargs)

@vllm_hook(
    hook_points="vllm.worker.model_runner.ModelRunner.execute_model",
    min_version="0.6.3"
)
def execute_model(original_func, this, model_input, kv_caches, *args, **kwargs):
    """模型执行钩子"""
    state = _get_state()

    if state.execute_model_first_run:
        state.execute_model_first_run = False
        return original_func(this, model_input, kv_caches, *args, **kwargs)

    prof = Profiler(Level.INFO).domain("ModelExecute")

    # 收集请求信息
    request_ids = [
        {"rid": rid}
        for rid in model_input.request_ids_to_seq_ids.keys()
    ]

    # 确定批次类型
    is_prefill = hasattr(model_input.attn_metadata, 'prefill_metadata') and \
                 model_input.attn_metadata.prefill_metadata is not None

    batch_type = "Prefill" if is_prefill else "Decode"

    # 设置分析器属性
    prof.res(request_ids)
    prof.attr("batch_type", batch_type)
    prof.attr("batch_size", model_input.input_tokens.shape[0])

    # 执行原始函数并记录耗时
    with prof.span("modelExec"):
        return original_func(this, model_input, kv_caches, *args, **kwargs)

@vllm_hook(
    hook_points="vllm.attention.backends.utils.CommonAttentionState.begin_forward",
    min_version="0.6.3"
)
def begin_forward(original_func, this, model_input, *args, **kwargs):
    """前向开始钩子"""
    state = _get_state()
    result = original_func(this, model_input, *args, **kwargs)

    if state.begin_forward_first_run:
        state.begin_forward_first_run = False
        return result

    # 创建新的性能分析器
    state.forward_profiler = Profiler(Level.INFO).domain("ModelExecute").res(
        [{"rid": rid} for rid in model_input.request_ids_to_seq_ids.keys()]
    )

    return result

@vllm_hook(
    hook_points="vllm.forward_context.set_forward_context",
    min_version="0.8.4"
)
@contextmanager
def set_forward_context(original_func, *args, **kwargs):
    """前向上下文钩子"""
    state = _get_state()
    prof = state.forward_profiler
    state.forward_profiler = None  # 重置状态

    # 执行原始上下文管理器
    with original_func(*args, **kwargs):
        if prof:
            with prof.span("forward"):
                yield
        else:
            yield

def cleanup_hooks():
    """清理所有钩子（测试/退出时使用）"""
    recover_hooks_for(handle_execute_model)
    recover_hooks_for(execute_model)
    recover_hooks_for(begin_forward)
    recover_hooks_for(set_forward_context)

# model_hookers = [ExecutorBaseExecuteModelHook, ModelRunnerExecuteHook, ModelForwardHook, SetForwardContextHook]
