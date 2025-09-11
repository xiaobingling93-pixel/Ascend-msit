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

from abc import abstractmethod
from typing import Optional, Set, Iterator, Tuple, Callable

from torch import nn


class WrapperIR(nn.Module):
    """
    包装类IR基类，用于包装其他nn.Module。
    
    该类是一个nn.Module，但持有一个被包装的nn.Module，提供统一的接口
    来访问和操作被包装的模块。
    """

    def __init__(self, wrapped_module: nn.Module):
        """
        初始化包装类IR。
        
        Args:
            wrapped_module: 被包装的nn.Module实例
        """
        super().__init__()
        self.wrapped_module = wrapped_module

    @staticmethod
    def is_atomic() -> bool:
        """
        如果该包装类IR是原子性的，则返回True，否则返回False。
        原子性包装类IR是指该IR应当被视为一个整体，不能被拆分。
        
        在保存时，原子性包装器会作为整体处理，不会单独处理被包装的模块。
        非原子性包装器会分别处理被包装模块和包装器自身。
        
        Returns:
            False，表示该包装类IR是非原子性的
        """
        return False

    def named_modules(
            self,
            memo: Optional[Set[nn.Module]] = None,
            prefix: str = '',
            remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, nn.Module]]:
        """
        重写named_modules方法，如果is_atomic()返回True，则只返回自身。
        
        Args:
            memo: 用于避免重复访问的模块集合
            prefix: 模块名称前缀
            remove_duplicate: 是否移除重复的模块
            
        Yields:
            模块名称和模块实例的元组
        """
        yield prefix, self


class HookIR(Callable):

    @abstractmethod
    def wrapper_module(self, module: nn.Module) -> WrapperIR:
        pass
