# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------
from typing import Dict

from auto_optimizer.common.register import Register
from auto_optimizer.common.utils import format_to_module


class ConfigDict(Dict):
    def __missing__(self, key):
        raise KeyError(key)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as err:
            raise AttributeError(f"'ConfigDict' object has no attribute '{item}'") from err


class Config:
    def __init__(self, config_dict=None):
        config_dict = config_dict or {}
        if not isinstance(config_dict, dict):
            raise TypeError('config_dict must be dict')

        super().__setattr__('_config_dict', ConfigDict(config_dict))

    def __repr__(self):
        return f'{self._config_dict.__repr__()}'

    def __len__(self):
        return len(self._config_dict)

    def __getattr__(self, item):
        return getattr(self._config_dict, item)

    def __getitem__(self, item):
        return self._config_dict[item]

    def __iter__(self):
        return iter(self._config_dict)

    def __getstate__(self):
        return self._config_dict

    @staticmethod
    def read_by_file(file_name):
        """
        读取模型配置文件，返回模型相关配置参数及推理流程
        """
        format_path = format_to_module(file_name)
        try:
            model_dict = Register.import_module(format_path)
        except Exception as err:
            raise RuntimeError("invalid read file error={}".format(err)) from err
        if not isinstance(model_dict.model, Dict):
            raise RuntimeError("config is not Dict")

        config_dict = model_dict.model

        return Config(config_dict)
