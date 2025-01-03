# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.

__all__ = [
    'replace_device_align_hook_if_needed',
    'move_update_weight_hook_if_need',
    'get_state_dict_copy',
    'clear_unused_module',
    'PrepareWeight',

    'enable_adapter',
    'disable_adapter',
    'check_model_compatible',
    'enabled_adapter',

    'get_offloaded_dataset',

    'copy_offloaded_state_dict',
    'DiskStateDictConfig',
    'MemoryStateDictConfig',

    'LazyTensor',
    'handle_lazy_tensor'
]

from .hook_adapter import (replace_device_align_hook_if_needed,
                           move_update_weight_hook_if_need,
                           get_state_dict_copy,
                           clear_unused_module,
                           PrepareWeight)
from .switch import check_model_compatible, enable_adapter, disable_adapter, enabled_adapter
from .utils import get_offloaded_dataset
from .offloaded_state_dict import (DiskStateDictConfig,
                                   MemoryStateDictConfig,
                                   copy_offloaded_state_dict)
from .lazy_handler import LazyTensor, handle_lazy_tensor
