# -*- coding: utf-8 -*-
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

from .logic import FunctionConstraint, IfElseConstraint
from ._path import (
    IsFile, Exists, IsDir, HasSoftLink, IsReadable, 
    IsWritable, IsExecutable, IsWritableToGroupOrOthers, 
    IsConsistentToCurrentUser, IsSizeReasonable
)
from .base import BaseConstraint


class PathConstraintBuilder:
    """
    Constraint builder for IDE support.

    Available constraints:
        is_file, file_exists, is_dir, has_soft_link, is_readable, is_writable,
        is_executable, is_not_writable_to_group_or_others, is_consistent_to_current_user,
        is_size_reasonable
    """
    @property
    def is_file(self): 
        return IsFile()
    
    @property
    def file_exists(self): 
        return Exists()
    
    @property
    def is_dir(self): 
        return IsDir()
    
    @property
    def has_soft_link(self): 
        return HasSoftLink()
    
    @property
    def is_readable(self): 
        return IsReadable()
    
    @property
    def is_writable(self): 
        return IsWritable()
    
    @property
    def is_executable(self): 
        return IsExecutable()
    
    @property
    def is_writable_to_group_or_others(self):
        return IsWritableToGroupOrOthers()
    
    @property
    def is_consistent_to_current_user(self):
        return IsConsistentToCurrentUser()
    
    @property
    def is_size_reasonable(self):
        return IsSizeReasonable()


path = PathConstraintBuilder()


def make_constraint(constraint, description=None):
    if isinstance(constraint, BaseConstraint):
        return constraint

    if callable(constraint):
        return FunctionConstraint(constraint, description)

    raise TypeError(
        f"Expected a BaseConstraint instance or a callable, but got {type(constraint).__name__}."
    )


def where(condition, if_constraint, else_constraint):
    return IfElseConstraint(condition, make_constraint(if_constraint), make_constraint(else_constraint))


READ_FILE_COMMON_CHECK = (
    path.is_file & ~path.has_soft_link & path.is_readable & 
    path.is_writable_to_group_or_others & path.is_consistent_to_current_user & path.is_size_reasonable
)
