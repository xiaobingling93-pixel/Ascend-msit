# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
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

from setuptools import setup

try:
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()
except Exception:
    long_description = "vLLM profiling tools plugin"

setup(
    name="vllm_profiler",
    version="0.1.0",
    author="Ascend MindStudio Toolkit team",
    description="vLLM profiling tools plugin",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="Apache-2.0",
    python_requires=">=3.8",
    install_requires=[
        "packaging>=21.0",
    ],
    packages=[
        "vllm_profiler",
        "vllm_profiler.vllm_v0",
        "vllm_profiler.vllm_v1",
    ],
    package_dir={
        "vllm_profiler": "vllm_profiler",
        "vllm_profiler.vllm_v0": "vllm_profiler/vllm_v0",
        "vllm_profiler.vllm_v1": "vllm_profiler/vllm_v1",
    },
    include_package_data=True,
    entry_points={
        "vllm.general_plugins": [
            "msserviceprofiler = vllm_profiler:register_service_profiler",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries",
    ],
)
