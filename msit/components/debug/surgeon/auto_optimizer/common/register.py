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
import os
import importlib

from auto_optimizer.common.utils import format_to_module
from components.debug.common import logger


class Register:
    def __init__(self, path_name: str):
        """
        path_name: 待注册文件夹的绝对路径
        """
        try:
            real_path = os.path.realpath(path_name)
        except Exception as err:
            raise RuntimeError("Invalid file error={}".format(err)) from err
        self.path_name = real_path

    @staticmethod
    def import_module(module):
        errors = []
        try:
            return importlib.import_module(module)
        except ImportError as error:
            errors.append((module, error))

        Register._handle_errors(errors)
        return None

    @staticmethod
    def _handle_errors(errors):
        if not errors:
            return

        for name, err in errors:
            raise RuntimeError("Module {} import failed: {}".format(name, err))

    def import_modules(self):
        modules = []
        try:
            self._add_modules(modules)
        except Exception as error:
            logger.error("add_modules failed, {}".format(error))
            raise RuntimeError("add_modules failed: {}".format(error)) from error

        for module in modules:
            if not module:
                continue
            Register.import_module(module)
        return True

    def _add_modules(self, modules: list):
        pwd_dir = os.getcwd()

        for root, _, files in os.walk(self.path_name, topdown=False):
            modules += [format_to_module(os.path.join(root.split(pwd_dir)[1], file)) for file in files]
