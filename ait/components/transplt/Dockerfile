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

FROM ubuntu:20.04

# Update Ubuntu repo to huaweicloud
RUN if [ "$(uname -m)" = "x86_64" ]; then \
        sed -i "s@http://.*archive.ubuntu.com@http://repo.huaweicloud.com@g" /etc/apt/sources.list \
        && sed -i "s@http://.*security.ubuntu.com@http://repo.huaweicloud.com@g" /etc/apt/sources.list; \
    elif [ "$(uname -m)" = "aarch64" ]; then \
        sed -i "s@http://ports.ubuntu.com@http://repo.huaweicloud.com@g" /etc/apt/sources.list \
        && sed -i "s@http://ports.ubuntu.com@http://repo.huaweicloud.com@g" /etc/apt/sources.list; \
    fi

# Update packages and install necessary dependencies
RUN apt-get update \
    && apt-get install -y wget gnupg software-properties-common

# Add the LLVM repository and install Clang-14 and Clang-14-dev using cernet repo
RUN wget -O - http://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - \
    && add-apt-repository "deb https://mirrors.cernet.edu.cn/llvm-apt/focal/ llvm-toolchain-focal-14 main" \
    && apt-get update \
    && apt-get install -y clang-14 libclang-14-dev

# Install Python 3.9 and pip and set pypi repo to huaweicloud
RUN apt-get install -y python3.9 python3.9-dev python3.9-distutils
RUN wget https://bootstrap.pypa.io/get-pip.py \
    && python3.9 get-pip.py \
    && rm get-pip.py \
    && pip3.9 config set global.index-url https://repo.huaweicloud.com/repository/pypi/simple

# Set Clang-14 as the default compiler
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 2 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 2 \
    && update-alternatives --install /usr/bin/pip3 pip3 /usr/local/bin/pip3.9 2

# Clone the repository
# Run the install.sh script
RUN apt-get install -y git \
    && git clone https://gitee.com/ascend/ait.git \
    && cd ait/ait \
    && chmod +x install.sh \
    && ./install.sh --transplt

# Go to the dist-packages directory
# Download and unzip config.zip, headers.zip
RUN cd /usr/local/lib/python3.9/dist-packages/app_analyze \
    && umask 022 \
    && apt-get install -y unzip \
    && wget -O config.zip https://ait-resources.obs.cn-south-1.myhuaweicloud.com/config.zip \
    && unzip config.zip \
    && rm config.zip \
    && wget -O headers.zip https://ait-resources.obs.cn-south-1.myhuaweicloud.com/headers.zip \
    && unzip headers.zip \
    && rm headers.zip

CMD ["bash"]
