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
import subprocess
import sys

from components.utils.constants import FileCheckConst
from components.utils.file_utils import FileChecker
from components.utils.install import AitInstaller


class LlmInstall(AitInstaller):
    @staticmethod
    def check():
        check_res = []

        if not os.path.exists(os.path.join(os.path.dirname(__file__), "opcheck/libopchecker.so")):
            check_res.append("[warnning] build libopchecker.so failed. will make the opchecker feature unusable. "
                             "use `msit build-extra llm` to try again")

        if not check_res:
            return "OK"
        else:
            return "\n".join(check_res)

    @staticmethod
    def build_extra(find_links=None):
        if sys.platform == 'win32':
            return

        if find_links is not None:
            file_check = FileChecker(find_links, FileCheckConst.DIR, ability=FileCheckConst.READ_WRITE_ABLE)
            file_check.common_check()
            os.environ['AIT_INSTALL_FIND_LINKS'] = file_check.file_path
        subprocess.run(
            ["/bin/bash", os.path.abspath(os.path.join(os.path.dirname(__file__), "install.sh"))], shell=False
        )

    @staticmethod
    def download_extra(dest):
        if sys.platform == 'win32':
            return

        file_check = FileChecker(dest, FileCheckConst.DIR, ability=FileCheckConst.READ_WRITE_ABLE)
        file_check.common_check()
        os.environ['AIT_DOWNLOAD_PATH'] = file_check.file_path
        subprocess.run(
            ["/bin/bash", os.path.abspath(os.path.join(os.path.dirname(__file__), "install.sh"))], shell=False
        )
