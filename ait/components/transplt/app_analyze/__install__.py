# Copyright (c) 2023-2023 Huawei Technologies Co., Ltd.
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
import os
import sys
import subprocess
from components.utils.install import AitInstaller


class TranspltInstall(AitInstaller):
    def check(self):
        check_res = []

        if not os.path.exists(os.path.join(os.path.dirname(__file__), "headers")):
            check_res.append("[error] download data failed. will make the transplt feature unusable. "
                             "use `ait build-extra transplt` to try again")
        from app_analyze.utils import clang_finder
        try:
            clang_finder.get_lib_clang_path()
        except RuntimeError as er:
            check_res.append(f"[error]{er} use `ait build-extra llvm` to try again")
        
        if not check_res:
            return "OK"
        else:
            return "\n".join(check_res)

    def build_extra(self, find_links=None):
        if find_links is not None:
            os.environ['AIT_INSTALL_FIND_LINKS'] = os.path.realpath(find_links)
        if sys.platform == 'win32':
            subprocess.run([os.path.join(os.path.dirname(__file__), "install.bat")])
        else:
            subprocess.run(["/bin/bash", os.path.join(os.path.dirname(__file__), "install.sh")])

    def download_extra(self, dest):
        os.environ['AIT_DOWNLOAD_PATH'] = os.path.realpath(dest)
        
        if sys.platform == 'win32':
            subprocess.run([os.path.join(os.path.dirname(__file__), "install.bat")]) 
        else:
            subprocess.run(["/bin/bash", os.path.join(os.path.dirname(__file__), "install.sh")])


class LlvmInstall(AitInstaller):
    def build_extra(self, find_links=None):
        if sys.platform == 'win32':
            subprocess.run([os.path.join(os.path.dirname(__file__), "install.bat"), "--full"])
        else:
            subprocess.run(["/bin/bash", os.path.join(os.path.dirname(__file__), "install.sh"), "--full"])
