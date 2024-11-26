#  Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import os
from typing import Mapping, Dict, Optional, Union

import torch
from accelerate.hooks import ModelHook, AlignDevicesHook, remove_hook_from_module, add_hook_to_module
from accelerate.utils import PrefixedDataset, OffloadedWeightsLoader
from accelerate.utils import offload_state_dict, offload_weight, save_offload_index, load_offloaded_weight
from accelerate.utils.memory import clear_device_cache
from safetensors import safe_open

from msmodelslim.pytorch.llm_ptq.accelerate_adapter.switch import enabled_adapter
from msmodelslim import logger as msmodelslim_logger


class PrepareWeight:
    """
    调用Accelerate的Hook，将模块的权重准备至执行设备上
    """

    def __init__(self, module: torch.nn.Module, post_force=False, post_recurse=False):

        self.module = module
        self.post_force = post_force
        self.post_recurse = post_recurse

    def __enter__(self):
        if not enabled_adapter():
            return

        if hasattr(self.module, '_hf_hook'):

            hook = getattr(self.module, '_hf_hook')

            # enable and save old state
            if isinstance(hook, UpdateWeightsMapHook):
                self.post_force = hook.enable_post_force(self.post_force)
                self.post_recurse = hook.enable_post_recurse(self.post_recurse)

            hook.pre_forward(self.module, *[], **{})

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not enabled_adapter():
            return

        if hasattr(self.module, '_hf_hook'):
            hook = getattr(self.module, '_hf_hook')

            hook.post_forward(self.module, *[torch.zeros([1])])
            # restore old state
            if isinstance(hook, UpdateWeightsMapHook):
                hook.enable_post_force(self.post_force)
                hook.enable_post_recurse(self.post_recurse)


def upload_module_weights(module: torch.nn.Module, post_force=False, post_recurse=False):
    if hasattr(module, '_hf_hook'):
        hook = getattr(module, '_hf_hook')

        if isinstance(hook, UpdateWeightsMapHook):
            hook.enable_post_force(post_force)
            hook.enable_post_recurse(post_recurse)

        hook.pre_forward(module, *[], **{})


def offload_module_weights(module: torch.nn.Module):
    if hasattr(module, '_hf_hook'):
        hook = getattr(module, '_hf_hook')
        hook.post_forward(module, *[torch.zeros([1])], **{})


class WritableOffloadedWeightsLoader(OffloadedWeightsLoader):
    """
    对OffloadedWeightsLoader的拓展，支持参数写入
    """

    def __init__(self,
                 state_dict: Dict[str, torch.Tensor] = None,
                 save_folder: Optional[Union[str, os.PathLike]] = None,
                 index: Mapping = None,
                 device=None):
        super().__init__(state_dict, save_folder, index, device)

    def update_all_keys(self):
        self.all_keys = list(self.state_dict.keys())
        self.all_keys.extend([key for key in self.index if key not in self.all_keys])
        pass

    def __setitem__(self, key, value):
        if key in self.state_dict or self.save_folder is None:
            self.state_dict[key] = value
            self.update_all_keys()
            return
        index = offload_weight(value, key, self.save_folder, index=self.index)
        save_offload_index(index, self.save_folder)
        offload_state_dict(self.save_folder, {key: value})
        self.update_all_keys()
        return

    def __getitem__(self, key: str):
        # State dict gets priority
        if key in self.state_dict:
            return self.state_dict[key]
        weight_info = self.index[key]
        if weight_info.get("safetensors_file") is not None:
            device = "cpu" if self.device is None else self.device
            tensor = None
            try:
                # device默认gpu触发CUDA初始化，此处约束为NPU
                with safe_open(weight_info["safetensors_file"], framework="pt", device=f"npu:{device}") as f:
                    tensor = f.get_tensor(weight_info.get("weight_name", key))
            except TypeError:
                # if failed to get_tensor on the device, such as bf16 on mps, try to load it on CPU first
                with safe_open(weight_info["safetensors_file"], framework="pt", device="cpu") as f:
                    tensor = f.get_tensor(weight_info.get("weight_name", key))

            if "dtype" in weight_info:
                tensor = tensor.to(getattr(torch, weight_info["dtype"]))

            if tensor.device != torch.device(device):
                tensor = tensor.to(device)
            return tensor

        weight_file = os.path.join(self.save_folder, f"{key}.dat")
        return load_offloaded_weight(weight_file, weight_info)


class UpdateWeightsMapHook(ModelHook):
    """
    包装AlignDevicesHook并拓展其功能，可以支持模块动态添加参数
    """

    logger = msmodelslim_logger

    def __init__(self, old_hook: AlignDevicesHook):
        if not isinstance(old_hook, AlignDevicesHook):
            raise ValueError("old_hook must be AlignDevicesHook")
        self.old_hook = old_hook
        self.init_force = False
        self.init_recurse = False
        self.post_force = False
        self.post_recurse = False
        self.depth = 0
        self.old_hook_offload = old_hook.offload

    def enable_init_force(self, enable: bool) -> bool:
        old_enable = self.init_force
        self.init_force = enable
        return old_enable

    def enable_init_recurse(self, enable: bool) -> bool:
        old_enable = self.init_recurse
        self.init_recurse = enable
        return old_enable

    def enable_post_force(self, enable: bool) -> bool:
        old_enable = self.post_force
        self.post_force = enable
        return old_enable

    def enable_post_recurse(self, enable: bool) -> bool:
        old_enable = self.post_recurse
        self.post_recurse = enable
        return old_enable

    def get_prefix(self):
        return self.old_hook.weights_map.prefix if self.old_hook.weights_map else "Unkown"

    def update_weights_map(self, module, force_update=False, recurse=False):

        """
        更新WritableOffloadedWeightsLoader的内容，分两种情况，一是参数值有更新，二是添加了新的参数。
        """

        if not self.old_hook_offload:
            return

        weights_map = self.old_hook.weights_map

        if weights_map is None:
            return

        if not isinstance(weights_map, PrefixedDataset):
            raise ValueError(f"weights_map {weights_map} is not PrefixedDataset")

        for key, item in module.named_parameters(recurse=recurse):
            if force_update or key not in weights_map:

                if key not in weights_map:
                    self.logger.debug(f"{weights_map.prefix + key} not in weight map, add it")
                else:
                    self.logger.debug(f"update {weights_map.prefix + key} in weight map")

                dataset = weights_map.dataset

                if not isinstance(dataset, WritableOffloadedWeightsLoader):
                    raise ValueError(f"dataset {dataset} must be WritableOffloadedWeightsLoader")

                dataset[weights_map.prefix + key] = item.clone().detach().cpu()

        clear_device_cache(True)

        return self

    def init_hook(self, module):
        self.depth = 0
        self.old_hook.offload = self.old_hook_offload
        self.update_weights_map(module, force_update=True, recurse=True)
        self.logger.debug(f"init_hook {self.get_prefix()} depth={self.depth}")
        return self.old_hook.init_hook(module)

    def detach_hook(self, module):
        self.depth = 0
        self.old_hook.offload = self.old_hook_offload
        self.logger.debug(f"detach_hook {self.get_prefix()} depth={self.depth}")
        return self.old_hook.detach_hook(module)

    def pre_forward(self, module, *args, **kwargs):

        self.depth += 1

        if self.depth > 1:
            # 对于嵌套的情况，不要再从WritableOffloadedWeightsLoader中再次加载
            self.old_hook.offload = False

        self.logger.debug(f"pre_forward {self.get_prefix()} depth={self.depth}")

        args, kwargs = self.old_hook.pre_forward(module, *args, **kwargs)

        return args, kwargs

    def post_forward(self, module, output):

        self.depth -= 1

        if self.depth == 0:
            # 对于嵌套的情况，所有嵌套都退出后，才更新权重
            self.old_hook.offload = self.old_hook_offload
            self.update_weights_map(module, force_update=self.post_force, recurse=self.post_recurse)

        self.logger.debug(f"post_forward {self.get_prefix()} depth={self.depth}")

        output = self.old_hook.post_forward(module, output)

        return output


def replace_offloaded_weights_loader_if_need(hook: AlignDevicesHook):
    if hook.weights_map is None:
        return

    if isinstance(hook.weights_map, PrefixedDataset):
        old_loader: OffloadedWeightsLoader = hook.weights_map.dataset
        new_loader = WritableOffloadedWeightsLoader(old_loader.state_dict, old_loader.save_folder, old_loader.index,
                                                    old_loader.device)
        hook.weights_map.dataset = new_loader

    return hook


def replace_device_align_hook_if_needed(module: torch.nn.Module, recurse=True, prefix=""):
    if not enabled_adapter():
        return

    msmodelslim_logger.debug(f"replace_device_align_hook_if_needed for {prefix}")

    if hasattr(module, '_hf_hook') and isinstance(getattr(module, '_hf_hook'), AlignDevicesHook):
        old_hook = getattr(module, '_hf_hook')
        replace_offloaded_weights_loader_if_need(old_hook)
        new_hook = UpdateWeightsMapHook(getattr(module, '_hf_hook'))
        setattr(module, '_hf_hook', new_hook)

    if not recurse:
        return

    for name, module in module.named_children():
        replace_device_align_hook_if_needed(module, recurse, prefix=f"{prefix}.{name}")


def move_update_weight_hook_if_need(old_module, new_module, as_submodule=False, force_update=False):
    """
    将old_module的hook移动至new_module，as_submodule设置为True时，old_module将会成为new_module的子模块
    """
    if not enabled_adapter():
        return

    if hasattr(old_module, '_hf_hook'):
        hook: UpdateWeightsMapHook = getattr(old_module, '_hf_hook')
        remove_hook_from_module(old_module)
        hook.old_hook.place_submodules = as_submodule
        old_init_force = hook.enable_init_force(force_update)
        old_init_recurse = hook.enable_init_recurse(as_submodule)
        add_hook_to_module(new_module, hook)
        hook.enable_init_force(old_init_force)
        hook.enable_init_recurse(old_init_recurse)


def get_offloaded_weights_loader_if_have(module, recurse=True) -> WritableOffloadedWeightsLoader:
    loader = None

    if hasattr(module, '_hf_hook'):
        hook = getattr(module, '_hf_hook')

        if isinstance(hook, UpdateWeightsMapHook):

            if hook.old_hook.weights_map is not None and isinstance(hook.old_hook.weights_map, PrefixedDataset):
                loader = hook.old_hook.weights_map.dataset

    if loader is not None:
        return loader

    if not recurse:
        return None

    for name, submodule in module.named_children():
        loader = get_offloaded_weights_loader_if_have(submodule, recurse)
        if loader is not None:
            return loader


def get_state_dict_copy(module: torch.nn.Module, skip_keys=None, device='cpu'):
    state_dict = {}
    weights_loader = get_offloaded_weights_loader_if_have(module)
    for key, value in module.state_dict().items():
        msmodelslim_logger.debug(f"{key} is on {value.device.type}")
        if skip_keys is not None and key in skip_keys:
            state_dict[key] = torch.zeros([1])
        elif value.device.type == 'meta':
            # load from model's offload weights loader
            state_dict[key] = weights_loader[key].clone().detach().to(device)
        else:
            state_dict[key] = value.clone().detach().to(device)
    return state_dict


def clear_unused_module(module: torch.nn.Module):
    if not enabled_adapter():
        return

    # 情况该模块的参数、子模块、Buffer
    module._parameters = {}
    module._named_modules = {}
    module._named_buffers = {}
