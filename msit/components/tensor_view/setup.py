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
from setuptools import setup, find_packages  # type: ignore

with open('requirements.txt', encoding='utf-8') as f:
    required = f.read().splitlines()

msit_sub_tasks = [
    {
    "name": "tensor-view",
    "help_info": "view、slice、permute、save the dumped tensor",
    "module": "ait_tensor_view.main_cli",
    "attr": "get_cmd_instance"
}
]

msit_sub_task_entry_points = [
    f"{t.get('name')}:{t.get('help_info')} = {t.get('module')}:{t.get('attr')}"
    for t in msit_sub_tasks
]

setup(
    name='msit-tensor-view',
    version='8.2.0',
    description='tensor view tool',
    long_description="Provides interfaces for viewing, slicing, transposing, and saving tensor",
    url='msit tensor-view url',
    packages=find_packages(),
    keywords='msit tensor-view tool',
    install_requires=required,
    python_requires='>=3.7',
    entry_points={
        'msit_sub_task': msit_sub_task_entry_points,
        'msit_sub_task_installer': ['msit-tensor-view=ait_tensor_view.__install__:TensorViewInstaller'],
    },
)
