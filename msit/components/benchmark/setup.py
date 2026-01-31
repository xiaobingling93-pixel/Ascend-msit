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
from configparser import ConfigParser
from setuptools import setup, find_packages  # type: ignore

config = ConfigParser()
cur_dir = os.path.dirname(os.path.abspath(__file__))  # benchmark
components_dir = os.path.dirname(cur_dir)  # components
config_ini_path = os.path.join(components_dir, 'config', 'config.ini')
config.read(config_ini_path)

msit_sub_tasks = [
    {
        "name": "benchmark",
        "help_info": "benchmark tool to get performance data including latency and throughput",
        "module": "msit_benchmark.__main__",
        "attr": "get_cmd_instance",
    }
]

msit_sub_task_entry_points = [
    f"{t.get('name')}:{t.get('help_info')} = {t.get('module')}:{t.get('attr')}" 
    for t in msit_sub_tasks
]

setup(
    name="msit-benchmark",
    version='8.2.0',
    description="msit benchmark tool",
    url=config.get('URL', 'msit_url'),
    packages=find_packages(),
    package_data={"": ["LICENSE", "README.md", "requirements.txt", "install.bat", "install.sh", "*.cpp", "*.h"]},
    keywords="msit benchmark tool",
    classifiers=[
        "Development Status :: Alpha",
        "Intended Audience :: Developers",
        'License :: OSI Approved :: Mulan Permissive Software License v2 (Mulan PSL v2)',
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development",
    ],
    python_requires=">=3.7",
    entry_points={
        "msit_sub_task": msit_sub_task_entry_points,
        "msit_sub_task_installer": ["msit-benchmark=msit_benchmark.__install__:BenchmarkInstall"],
    },
)
