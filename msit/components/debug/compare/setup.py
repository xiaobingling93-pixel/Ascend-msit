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
from configparser import ConfigParser
from setuptools import setup, find_packages  # type: ignore

config = ConfigParser()
config.read('../../config/config.ini')

with open('requirements.txt', encoding='utf-8') as f:
    required = f.read().splitlines()

debug_sub_tasks = [{
    "name": "compare",
    "help_info": "one-click network-wide accuracy analysis of golden models.",
    "module": "msquickcmp.__main__",
    "attr": "get_compare_cmd_ins"
}, {
    "name": "dump",
    "help_info": "one-click dump model ops inputs and outputs.",
    "module": "msquickcmp.__main__",
    "attr": "get_dump_cmd_ins"
}]

debug_sub_task_entry_points = [
    f"{t.get('name')}:{t.get('help_info')} = {t.get('module')}:{t.get('attr')}"
    for t in debug_sub_tasks
]

setup(
    name='msit-compare',
    version='8.2.0',
    description='This tool enables one-click network-wide accuracy analysis of gold model.',
    long_description="",
    long_description_content_type='text/markdown',
    url=config.get('URL', 'msit_debug_compare_url'),
    packages=find_packages(),
    package_data={'': ['LICENSE', 'install.sh', 'libsaveom.so', '*.cpp']},
    license='Mulan PSL v2',
    keywords='compare',
    install_requires=required,
    classifiers=[
        'Development Status :: Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Mulan Permissive Software License v2 (Mulan PSL v2)',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development'
    ],
    python_requires='>=3.7',
    entry_points={
        'debug_sub_task': debug_sub_task_entry_points,
        'msit_sub_task_installer': ['msit-compare=msquickcmp.__install__:CompareInstall'],
    },
)
