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
import sys
import subprocess
import pkg_resources
from components.utils.install import AitInstaller


class BenchmarkInstall(AitInstaller):
    @staticmethod
    def check():
        check_res = []
        installed_pkg = [pkg.key for pkg in pkg_resources.working_set]

        if "aclruntime" not in installed_pkg:
            check_res.append("[error] aclruntime not installed. use `msit build-extra benchmark` to try again")

        if "ais-bench" not in installed_pkg:
            check_res.append("[error] ais-bench not installed. use `msit build-extra benchmark` to try again")

        if not check_res:
            return "OK"
        else:
            return "\n".join(check_res)

    @staticmethod
    def build_extra(find_links=None):
        if sys.platform == "win32":
            return

        if find_links is not None:
            os.environ["MSIT_INSTALL_FIND_LINKS"] = os.path.realpath(find_links)
        subprocess.run(["/bin/bash", os.path.abspath(os.path.join(os.path.dirname(__file__), "install.sh"))])

    @staticmethod
    def download_extra(dest):
        if sys.platform == "win32":
            return

        os.environ["MSIT_DOWNLOAD_PATH"] = os.path.realpath(dest)
        subprocess.run(["/bin/bash", os.path.abspath(os.path.join(os.path.dirname(__file__), "install.sh"))])
