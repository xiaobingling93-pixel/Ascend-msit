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
import site
import os

from configparser import ConfigParser
from setuptools import setup, find_packages

config = ConfigParser()
config.read('../config/config.ini')

with open('requirements.txt', encoding='utf-8') as f:
    required = f.read().splitlines()

opchecker_lib_src = []
for root, _, files in os.walk('components/llm/msit_llm/opcheck/test_framework/'):
    opchecker_lib_src.append((os.path.join("/", root), [os.path.join(root, f) for f in files]))

msit_sub_tasks = [
    {
        "name": "llm",
        "help_info": "Large Language Model(llm) Debugger Tools.",
        "module": "msit_llm.__main__",
        "attr": "get_cmd_instance"
    }
]

msit_sub_task_entry_points = [
    f"{t.get('name')}:{t.get('help_info')} = {t.get('module')}:{t.get('attr')}"
    for t in msit_sub_tasks
]

if not site.getsitepackages():
    raise RuntimeError('There are no global site-packages directories.')

setup(
    name='msit_llm',
    version='8.2.0',
    description='Debug tools for large language model(llm)',
    url=config.get('URL', 'msit_llm_url'),
    packages=find_packages(),
    package_data={'': ['*.sh', '*.cpp', '*.h', '*.txt']},
    license='Mulan PSL v2',
    keywords='msit_llm',
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
    data_dir=f"{site.getsitepackages()[0]}",
    data_files=opchecker_lib_src,
    include_package_data=True,
    python_requires='>=3.7',
    entry_points={
        'msit_sub_task': msit_sub_task_entry_points,
        'msit_sub_task_installer': ['msit-llm=msit_llm.__install__:LlmInstall'],
    },
)
