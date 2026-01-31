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
import argparse
import os

from components.utils.constants import AIT_FAQ_HOME, MIND_STUDIO_LOGO
from components.utils.file_utils import root_privilege_warning
from components.utils.parser import (
    BaseCommand,
    AitInstallCommand,
    AitBuildExtraCommand,
    AitCheckCommand,
    DownloadCommand
)


class UmaskWrapper:
    """Write with preset umask
    >>> with UmaskWrapper():
    >>>     ...
    """

    def __init__(self, umask=0o027):
        self.umask, self.ori_umask = umask, None

    def __enter__(self):
        self.ori_umask = os.umask(self.umask)

    def __exit__(self, exc_type=None, exc_val=None, exc_tb=None):
        os.umask(self.ori_umask)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=f"msit(MindStudio Inference Tools), {MIND_STUDIO_LOGO}.\n"
                    "Providing one-site debugging and optimization toolkit for inference on Ascend Devices.\n"
                    f"For any issue, refer FAQ first: {AIT_FAQ_HOME}",
    )

    cmd = BaseCommand(
        "msit", None, [
            "msit_sub_task",
            AitInstallCommand(),
            AitBuildExtraCommand(),
            AitCheckCommand(),
            DownloadCommand()
        ]
    )

    cmd.register_parser(parser)

    args = parser.parse_args()

    if hasattr(args, 'handle'):
        with UmaskWrapper():
            root_privilege_warning()
            try:
                args.handle(args)
            except Exception as err:
                raise Exception(
                    f"[ERROR] Refer FAQ if a known issue: {AIT_FAQ_HOME}"
                ) from err


if __name__ == "__main__":
    main()
