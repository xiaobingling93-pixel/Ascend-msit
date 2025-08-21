#  -*- coding: utf-8 -*-
#  -*- coding: utf-8 -*-
#  Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#  http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


from typing import Any, Optional

import torch
from torch import nn

from msmodelslim.utils.logging import get_logger

MAX_RECURSION_DEPTH = 10


def _align_input_to_module_device(module: Optional[nn.Module], input_data: Any) -> Any:
    """
    将模块的输入数据对齐到模块所在的设备上的内部函数。

    参数:
        module: 目标模块
        input_data: 输入数据，通常是 (args, kwargs) 的元组

    返回:
        对齐后的输入数据
    """
    if module is None or input_data is None:
        return input_data

    # 获取模块所在的设备
    try:
        module_device = next(module.parameters()).device
    except StopIteration:
        # 如果模块没有参数，返回原始输入
        return input_data

    # 统计移动的数据总量
    total_moved_bytes = 0
    moved_tensors_count = 0

    def _to_device(data: Any, device: torch.device, depth: int = 0) -> Any:
        """递归地将数据移动到指定设备"""
        nonlocal total_moved_bytes, moved_tensors_count
        if depth > MAX_RECURSION_DEPTH:
            raise RecursionError(f"Maximum recursion depth {MAX_RECURSION_DEPTH} exceeded")

        if isinstance(data, dict):
            return {k: _to_device(v, device, depth=depth + 1) for k, v in data.items()}
        elif isinstance(data, list):
            return [_to_device(item, device, depth=depth + 1) for item in data]
        elif isinstance(data, tuple):
            return tuple(_to_device(item, device, depth=depth + 1) for item in data)
        elif isinstance(data, torch.Tensor):
            # 检查张量是否已经在目标设备上
            if data.device != device:
                # 统计移动的数据量
                tensor_bytes = data.element_size() * data.numel()
                total_moved_bytes += tensor_bytes
                moved_tensors_count += 1
                return data.to(device)
            else:
                return data
        else:
            return data

    result = _to_device(input_data, module_device)

    # 在 debug 日志中提示移动的数据总量
    if moved_tensors_count > 0:
        get_logger().debug(
            "Device alignment hook for %r: moved %d tensors to device %r, total size: %r, "
            "device memory: allocated=%r, reserved=%r",
            module.__class__.__name__,
            moved_tensors_count,
            module_device.type,
            format_memory_size(total_moved_bytes),
            format_memory_size(get_device_allocated_memory()),
            format_memory_size(get_device_reserved_memory())
        )
    else:
        get_logger().debug(
            "Device alignment hook for %r: no tensors moved (already on device %r), "
            "device memory: allocated=%r, reserved=%r",
            module.__class__.__name__,
            module_device.type,
            format_memory_size(get_device_allocated_memory()),
            format_memory_size(get_device_reserved_memory())
        )

    return result


def align_input_to_module_device_hook(module: Optional[nn.Module], args: Any, kwargs: Any) -> Any:
    """
    将模块的输入数据对齐到模块所在的设备上的 hook 函数。

    该函数作为 forward pre-hook 使用，在模块 forward 前自动将输入数据移动到模块所在设备。

    参数:
        module: 目标模块
        *args: 位置参数（当with_kwargs=False时，第一个参数是input_data）
        **kwargs: 关键字参数

    返回:
        对齐后的输入数据
    """
    input_data = (args, kwargs,)
    aligned_input_data = _align_input_to_module_device(module, input_data)
    return aligned_input_data[0], aligned_input_data[1]


def register_device_alignment_hook(module: Optional[nn.Module], with_kwargs: bool = False,
                                   name: Optional[str] = None) -> Optional[
    torch.utils.hooks.RemovableHandle]:
    """
    为模块注册设备对齐 hook。

    该函数会为指定的模块注册一个 forward pre-hook，在每次 forward 前自动将输入数据
    对齐到模块所在的设备上。

    参数:
        module: 需要注册 hook 的模块
        with_kwargs: 是否使用kwargs格式的hook，默认为False
        name: 模块名称，用于日志输出，默认为None

    返回:
        hook handle: 注册的 hook 句柄，用于后续移除 hook
    """
    if module is None:
        return None

    # 检查是否已经注册过 hook
    if hasattr(module, '_device_alignment_hook_registered'):
        return module._device_alignment_hook_handle

    # 注册 forward pre-hook
    hook_handle = module.register_forward_pre_hook(align_input_to_module_device_hook, with_kwargs=with_kwargs)

    # 标记已注册 hook
    module._device_alignment_hook_registered = True
    module._device_alignment_hook_handle = hook_handle

    # 使用提供的name或模块类名进行日志输出
    module_name = name if name is not None else module.__class__.__name__
    get_logger().debug("Registered device alignment hook for %r", module_name)

    return hook_handle


def unregister_device_alignment_hook(module: Optional[nn.Module], name: Optional[str] = None) -> None:
    """
    移除模块的设备对齐 hook。

    参数:
        module: 需要移除 hook 的模块
        name: 模块名称，用于日志输出，默认为None
    """
    if module is None or not hasattr(module, '_device_alignment_hook_handle'):
        return

    # 移除 hook
    module._device_alignment_hook_handle.remove()

    # 清除标记
    delattr(module, '_device_alignment_hook_registered')
    delattr(module, '_device_alignment_hook_handle')

    # 使用提供的name或模块类名进行日志输出
    module_name = name if name is not None else module.__class__.__name__
    get_logger().debug("Unregistered device alignment hook for %r", module_name)


def get_module_param_size(module: nn.Module) -> int:
    """
    计算模块的参数大小（字节数）

    Args:
        module: 待计算的模块

    Returns:
        模块参数总大小（字节数）
    """
    total_size: int = 0
    for param in module.parameters():
        total_size += param.numel() * param.element_size()
    return total_size


def get_device_allocated_memory(device_id: int = 0) -> int:
    """
    获取设备已分配内存大小，自动检测设备类型

    Args:
        device_id: 设备ID，默认为0

    Returns:
        已分配内存大小，单位为字节
    """
    # 自动检测设备类型
    if hasattr(torch, 'cuda') and torch.cuda.is_available():
        return torch.cuda.memory_allocated(device_id)
    elif hasattr(torch, 'npu') and torch.npu.is_available():
        return torch.npu.memory_allocated(device_id)
    return 0


def get_device_reserved_memory(device_id: int = 0) -> int:
    """
    获取设备已申请内存大小，自动检测设备类型

    Args:
        device_id: 设备ID，默认为0

    Returns:
        已申请内存大小，单位为字节
    """
    # 自动检测设备类型
    if hasattr(torch, 'npu') and torch.npu.is_available():
        return torch.npu.memory_reserved(device_id)
    elif hasattr(torch, 'cuda') and torch.cuda.is_available():
        return torch.cuda.memory_reserved(device_id)
    return 0


def format_memory_size(size_bytes: int) -> str:
    """
    格式化内存大小显示

    Args:
        size_bytes: 字节数

    Returns:
        格式化后的内存大小字符串
    """
    if size_bytes < 1024:
        return "%dB" % size_bytes
    elif size_bytes < 1024 * 1024:
        return "%.2fKB" % (size_bytes / 1024)
    elif size_bytes < 1024 * 1024 * 1024:
        return "%.2fMB" % (size_bytes / (1024 * 1024))
    else:
        return "%.2fGB" % (size_bytes / (1024 * 1024 * 1024))
