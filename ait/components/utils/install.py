# Copyright (c) 2023-2023 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import logging
import sys
import argparse
import subprocess
from typing import Union
from components.utils.util import get_entry_points
from components.utils.parser import BaseCommand

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO, format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def is_windows():
    return sys.platform == "win32"


def warning_in_windows(title):
    if is_windows():
        logger.warning(f"{title} is not support windows")
        return True
    return False


def get_base_path():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    print(base)
    return base


def get_real_pkg_path(pkg_path):
    return os.path.join(get_base_path(), pkg_path)


class AitInstaller:
    def check(self):
        return "OK"

    def build_extra(self, find_links):
        logger.info("there are no more extra dependencies to build")

    def download_extra(self, dest):
        logger.info("there are no more extra dependencies to download")


INSTALL_INFO_MAP = [
    {
        "arg-name": "llm",
        "pkg-name": "ait-llm",
        "pkg-path": "llm",
    },
    {
        "arg-name": "surgeon",
        "pkg-name": "ait-surgeon",
        "pkg-path": os.path.join("debug", "surgeon"),
        "support_windows": True,
    },
    {
        "arg-name": "analyze",
        "pkg-name": "ait-analyze",
        "pkg-path": "analyze",
    },
    {
        "arg-name": "transplt",
        "pkg-name": "ait-transplt", 
        "pkg-path": "transplt", 
        "support_windows": True},
    {
        "arg-name": "convert",
        "pkg-name": "ait-convert",
        "pkg-path": "convert",
    },
    {
        "arg-name": "profile",
        "pkg-name": "ait-profile",
        "pkg-path": os.path.join("profile", "msprof"),
    },
    {
        "arg-name": "tensor-view",
        "pkg-name": "ait-tensor-view",
        "pkg-path": "tensor_view"
    },
    {
        "arg-name": "benchmark",
        "pkg-name": "msit-benchmark",
        "pkg-path": "benchmark",
    },
    {
        "arg-name": "compare",
        "pkg-name": "ait-compare",
        "pkg-path": os.path.join("debug", "compare"),
        "depends": ["msit-benchmark", "ait-surgeon"],
    },
]

ALL_SUB_TOOLS = [pkg.get("arg-name") for pkg in INSTALL_INFO_MAP]
ALL_SUB_TOOLS_WITH_ALL = ["all"]
ALL_SUB_TOOLS_WITH_ALL.extend(ALL_SUB_TOOLS)


class AitInstallCommand(BaseCommand):
    def __init__(self) -> None:
        super().__init__("install", "install ait tools", group="Install Command")

    def add_arguments(self, parser: argparse.ArgumentParser):
        parser.add_argument(
            "comp_names",
            default=None,
            nargs="+",
            choices=ALL_SUB_TOOLS_WITH_ALL,
            help="component's name",
        )

        parser.add_argument(
            "--find-links", "-f",
            default=None,
            type=str,
            help="the dir look for archives",
        )

    def handle(self, args):
        install_tools(args.comp_names, args.find_links)


class AitCheckCommand(BaseCommand):
    def __init__(self) -> None:
        super().__init__("check", "check ait tools status.", group="Install Command")

    def add_arguments(self, parser: argparse.ArgumentParser):
        parser.add_argument(
            "comp_names",
            default=None,
            nargs="+",
            choices=ALL_SUB_TOOLS_WITH_ALL,
            help="component's name",
        )

    def handle(self, args):
        check_tools(args.comp_names)


class AitBuildExtraCommand(BaseCommand):
    def __init__(self) -> None:
        super().__init__("build-extra", "build ait tools extra", group="Install Command")

    def add_arguments(self, parser: argparse.ArgumentParser):
        parser.add_argument(
            "comp_name",
            default=None,
            choices=ALL_SUB_TOOLS,
            help="component's name",
        )

        parser.add_argument(
            "--find-links", "-f",
            default=None,
            type=str,
            help="the dir look for archives",
        )

    def handle(self, args):
        build_extra(args.comp_name, args.find_links)


class DownloadCommand(BaseCommand):
    def __init__(self) -> None:
        super().__init__("download", "download packages", group="Install Command")

    def add_arguments(self, parser: argparse.ArgumentParser):
        parser.add_argument(
            "comp_names",
            default=None,
            choices=ALL_SUB_TOOLS_WITH_ALL,
            help="component's name",
        )

        parser.add_argument(
            "--dest", "-d",
            default=None,
            required=True,
            type=str,
            help=" Download packages into <dir>.",
        )

    def handle(self, args):
        download_comps(args.comp_names, args.dest)



def get_install_info_follow_depends(install_infos):
    all_names = set()
    for info in install_infos:
        all_names.add(info.get("pkg-name"))
        all_names.update(info.get("depends", []))
    if len(all_names) == len(install_infos):
        return install_infos
    else:
        return list(
            filter(lambda info: info["pkg-name"] in all_names, INSTALL_INFO_MAP)
        )


def install_tools(names, find_links):
    if names is None or len(names) == 0:
        logger.info(
            "You can specify the components you want to install, "
            "you can select more than one, "
            "or you can use install all to install all components."
        )
        return
    if "all" in names:
        install_infos = INSTALL_INFO_MAP
    else:
        install_infos = list(
            filter(lambda info: info["arg-name"] in names, INSTALL_INFO_MAP)
        )

        install_infos = get_install_info_follow_depends(install_infos)

    for tool_info in install_infos:
        install_tool(tool_info, find_links)


def install_tool(tool_info, find_links):
    pkg_name = tool_info.get("pkg-name")
    arg_name = tool_info.get("arg-name")
    support_windows = tool_info.get("support_windows", False)
    if not support_windows and warning_in_windows(pkg_name):
        return
    logger.info(f"installing {pkg_name}")
    pkg_path = get_real_pkg_path(tool_info.get("pkg-path"))

    if find_links is not None:
        subprocess.run([sys.executable, "-m", "pip", "install", pkg_path, "--no-index", "-f", find_links])
        subprocess.run([sys.executable, "-m", "components", "build-extra", arg_name, "-f", find_links])
    else:
        subprocess.run([sys.executable, "-m", "pip", "install", pkg_path])
        subprocess.run([sys.executable, "-m", "components", "build-extra", arg_name])


def get_installer(pkg_name) -> Union[AitInstaller, None]:
    entry_points = get_entry_points("ait_sub_task_installer")
    pkg_installer = None
    for entry_point in entry_points:
        if entry_point.name == pkg_name:
            pkg_installer = entry_point.load()()
            break
    if isinstance(pkg_installer, AitInstaller):
        return pkg_installer
    return None


def check_tools(names):
    if names is None or "all" in names or len(names) == 0:
        install_infos = INSTALL_INFO_MAP
    else:
        install_infos = filter(lambda info: info["arg-name"] in names, INSTALL_INFO_MAP)

    for tool_info in install_infos:
        pkg_name = tool_info.get("pkg-name")
        logger.info(pkg_name)
        for msg in check_tool(pkg_name).split("\n"):
            logger.info(f"  {msg}")


def check_tool(pkg_name):
    logger.debug(f"checking {pkg_name}")
    pkg_installer = get_installer(pkg_name)

    if not pkg_installer:
        return "not install yet."
    else:
        return pkg_installer.check()


def build_extra(name, find_links):
    pkg_name = None

    for pkg_info in INSTALL_INFO_MAP:
        if pkg_info.get("arg-name") == name:
            pkg_name = pkg_info.get("pkg-name")
            break

    if pkg_name is None:
        raise ValueError("unknow error, pkg_name not found.")
    logger.info(f"building extra of {pkg_name}")
    pkg_installer = get_installer(pkg_name)

    if not pkg_installer:
        pkg_installer = AitInstaller()
    return pkg_installer.build_extra(find_links)


def download_comps(names, dest):
    if names is None or "all" in names or len(names) == 0:
        install_infos = INSTALL_INFO_MAP
    else:
        install_infos = filter(lambda info: info["arg-name"] in names, INSTALL_INFO_MAP)
   
    install_infos = get_install_info_follow_depends(list(install_infos))

    for tool_info in install_infos:
        download_comp(tool_info, dest)
    return install_infos


def download_comp(tool_info, dest):
    pkg_name = tool_info.get("pkg-name")
    support_windows = tool_info.get("support_windows", False)
    if not support_windows and warning_in_windows(pkg_name):
        return 
    logger.info(f"installing {pkg_name}")
    pkg_path = get_real_pkg_path(tool_info.get("pkg-path"))

    subprocess.run([sys.executable, "-m", "pip", "download", "-d", dest, pkg_path], shell=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "-f", dest, pkg_path], shell=False)
    
    pkg_installer = get_installer(pkg_name)

    if not pkg_installer:
        pkg_installer = AitInstaller()
    pkg_installer.download_extra(dest)
