# Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
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

import argparse
import os
from components.utils.parser import BaseCommand
from app_analyze.utils import log_util
from app_analyze.porting.app import start_scan_kit
from components.utils.file_open_check import FileStat



def check_source_path(value):
    source_list = value.split(',')
    for path in source_list:
        path_value = str(path)
        try:
            file_stat = FileStat(path_value)
        except Exception as err:
            raise argparse.ArgumentTypeError(f"source path:{path_value} is illegal. Please check.") from err
        if not file_stat.is_basically_legal('read'):
            raise argparse.ArgumentTypeError(f"source path:{path_value} is illegal. Please check.")
        if not file_stat.is_dir:
            raise argparse.ArgumentTypeError(f"source path:{path_value} is not a directory. Please check.")
    return value


class TranspltCommand(BaseCommand):
    def add_arguments(self, parser):
        # 逗号分隔的情况下只有一个列表元素
        parser.add_argument(
            "-s", "--source", type=check_source_path, required=True, help="directories of source folder"
        )
        parser.add_argument(
            "-f",
            "--report-type",
            default='csv',
            choices=['csv', 'json'],
            help="specify output report type. Only csv(xlsx)/json is supported",
        )
        parser.add_argument(
            "--log-level", default="INFO", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help="specify log level"
        )
        parser.add_argument(
            "--tools", default="cmake", choices=['cmake', 'python'],
            help="specify construction, currently support cmake and python"
        )
        parser.add_argument(
            "--mode", default="all", choices=['all', 'api-only'],
            help="specify scanner mode, currently support all and api only"
        )

    @staticmethod
    def _set_env():
        if os.path.exists("/opt/rh/llvm-toolset-7.0/root/usr/lib64/clang/7.0.1/include"):
            extra_path = "/opt/rh/llvm-toolset-7.0/root/usr/lib64/clang/7.0.1/include"
        elif os.path.exists("/usr/lib64/clang/7.0.1/include"):
            extra_path = "/usr/lib64/clang/7.0.1/include"
        else:
            extra_path = ""

        c_plus_include_path = os.environ.get("CPLUS_INCLUDE_PATH")
        if len(extra_path) > 0:
            c_plus_include_path = f"{extra_path}:{c_plus_include_path}"
            os.environ["CPLUS_INCLUDE_PATH"] = c_plus_include_path

    def handle(self, args):
        log_util.set_logger_level(args.log_level)
        log_util.init_file_logger()
        self._set_env()
        start_scan_kit(args)


def get_cmd_instance():
    help_info = "Transplant tool to analyze inference applications"
    cmd_instance = TranspltCommand("transplt", help_info)
    return cmd_instance
