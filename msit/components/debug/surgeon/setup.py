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



debug_sub_tasks = [
    {
        "name": "surgeon",
        "help_info": "surgeon tool for onnx modifying functions.",
        "module": "auto_optimizer.ait_main",
        "attr": "get_cmd_instance"
    }
]

debug_sub_task_entry_points = [
    f"{t.get('name')}:{t.get('help_info')} = {t.get('module')}:{t.get('attr')}"
    for t in debug_sub_tasks
]

setup(
    name='msit-surgeon',
    version='8.2.0',
    description='auto optimizer',
    long_description="",
    long_description_content_type='text/markdown',
    url=config.get('URL', 'msit_url'),
    packages=find_packages(),
    package_data={'': ['LICENSE', 'model.cfg']},    

    license='Mulan PSL v2',
    keywords='auto optimizer',
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
        'Topic :: Software Development',
    ],
    extras_require={
        'inference': [
            (
                f"aclruntime @ git+{config.get('URL', 'msit_url')}"
                '#egg=aclruntime&subdirectory=msit/components/benchmark/backend'
            ),
            (
                f"ais_bench @ git+{config.get('URL', 'msit_url')}"
                '#egg=ais_bench&subdirectory=msit/components/benchmark/'
            ),
            'pillow >= 9.0.0',
            'tqdm >= 4.63.0',
        ],
        'simplify': ['onnx-simplifier >= 0.3.6'],
    },
    python_requires='>=3.7',
    entry_points={
        'console_scripts': ['auto_optimizer=auto_optimizer.__main__:cli'],
        'debug_sub_task': debug_sub_task_entry_points,
        'msit_sub_task_installer': ['msit-surgeon=auto_optimizer.__install__:SurgeonInstall'],
    },
)
