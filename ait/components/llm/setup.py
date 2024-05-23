# Copyright (c) 2023-2023 Huawei Technologies Co., Ltd.
#
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

import subprocess
import site
import os

from setuptools import setup, find_packages

with open('requirements.txt', encoding='utf-8') as f:
    required = f.read().splitlines()

opchecker_lib_src = []
for root, dirs, files in os.walk('components/llm/ait_llm/opcheck/test_framework/'):
    opchecker_lib_src.append((os.path.join("/", root), [os.path.join(root, f) for f in files]))

ait_sub_tasks = [{
    "name": "llm",
    "help_info": "Large Language Model(llm) Debugger Tools.",
    "module": "ait_llm.__main__",
    "attr": "get_cmd_instance"
}]

ait_sub_task_entry_points = [
    f"{t.get('name')}:{t.get('help_info')} = {t.get('module')}:{t.get('attr')}"
    for t in ait_sub_tasks
]

setup(
    name='ait-llm',
    version='7.0.0c2',
    description='Debug tools for large language model(llm)',
    url='https://gitee.com/ascend/ait/ait/components/llm',
    packages=find_packages(),    
    package_data={'': ['*.sh', '*.cpp', '*.h', '*.txt']},
    license='Apache-2.0',
    keywords='ait_llm',
    install_requires=required,
    classifiers=[
        'Development Status :: Alpha',
        'Intended Audience :: Developers',
        'License :: Apache-2.0 Software License',
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
        'ait_sub_task': ait_sub_task_entry_points,
        'ait_sub_task_installer': ['ait-llm=ait_llm.__install__:LlmInstall'],
    },
)