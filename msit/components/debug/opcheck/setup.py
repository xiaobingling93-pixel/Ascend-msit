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
    "name": "opcheck",
    "help_info": "Operation check tool for GE compile model.",
    "module": "msit_opcheck.__main__",
    "attr": "get_opcheck_cmd_ins"
}]

debug_sub_task_entry_points = [
    f"{t.get('name')}:{t.get('help_info')} = {t.get('module')}:{t.get('attr')}"
    for t in debug_sub_tasks
]

setup(
    name='msit-opcheck',
    version='8.2.0',
    description='This tool enables operation check tool for GE compile model.',
    long_description="",
    url=config.get('URL', 'msit_debug_opcheck_url'),
    packages=find_packages(),
    package_data={'': ['LICENSE', 'install.sh']},
    license='Mulan PSL v2',
    keywords='opcheck',
    install_requires=required,
    classifiers=[
        'Development Status :: Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Mulan Permissive Software License v2 (Mulan PSL v2)',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development'
    ],
    python_requires='>=3.6',
    entry_points={
        'debug_sub_task': debug_sub_task_entry_points,
        'msit_sub_task_installer': ['msit-opcheck=msit_opcheck.__install__:OpCheckInstall'],
    },
)
