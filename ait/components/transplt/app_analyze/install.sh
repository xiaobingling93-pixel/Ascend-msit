#!/usr/bin/env bash
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

SUDO=""
type sudo > /dev/null 2>&1
ret=$?
if [ $ret -eq 0 ]; then
  SUDO="sudo"
fi

OS_NAME=$(grep -E "^NAME=" /etc/os-release | cut -d'=' -f2 | tr -d '"')
OS_VERSION=$(grep -E "^VERSION=" /etc/os-release | cut -d'=' -f2 | cut -d' ' -f1 | sed 's/\"//g')

install_clang_on_ubuntu() {
  if [ "$(uname -m)" = "x86_64" ]; then \
      sed -i "s@http://.*archive.ubuntu.com@http://repo.huaweicloud.com@g" /etc/apt/sources.list \
      && sed -i "s@http://.*security.ubuntu.com@http://repo.huaweicloud.com@g" /etc/apt/sources.list; \
  elif [ "$(uname -m)" = "aarch64" ]; then \
      sed -i "s@http://ports.ubuntu.com@http://repo.huaweicloud.com@g" /etc/apt/sources.list \
      && sed -i "s@http://ports.ubuntu.com@http://repo.huaweicloud.com@g" /etc/apt/sources.list; \
  fi

  $SUDO apt-get update && $SUDO apt-get install -y wget gnupg software-properties-common

  if [[ $OS_VERSION == "22.04"* ]] || [[ $OS_VERSION == "20.04"* ]]; then
    wget -O - http://apt.llvm.org/llvm-snapshot.gpg.key | $SUDO apt-key add -
    $SUDO echo "deb https://mirrors.cernet.edu.cn/llvm-apt/focal/ llvm-toolchain-focal-14 main" >> /etc/apt/sources.list
    $SUDO apt-get update
    $SUDO apt-get install libclang-14-dev clang-14 -y
  elif [[ $OS_VERSION == "18.04"* ]]; then
    $SUDO apt-get update
    $SUDO apt-get install libclang-10-dev clang-10 -y
  elif [[ $OS_VERSION == "16.04"* ]]; then
    $SUDO apt-get update
    $SUDO apt-get install libclang-6.0-dev clang-6.0 -y
  fi
}

install_clang_on_centos() {
  $SUDO yum install centos-release-scl-rh -y
  $SUDO yum install llvm-toolset-7.0-clang -y
  source /opt/rh/llvm-toolset-7.0/enable
  echo "source /opt/rh/llvm-toolset-7.0/enable" >> ~/.bashrc
}

install_clang_on_sles() {
  $SUDO zypper addrepo -f http://mirrors.163.com/openSUSE/update/leap/15.1/non-oss update-repo-no-oss163
  $SUDO zypper addrepo -f http://mirrors.163.com/openSUSE/update/leap/15.1/oss update-repo-oss163
  $SUDO zypper addrepo -f http://mirrors.163.com/openSUSE/distribution/leap/15.1/repo/oss dis-repo-oss163
  $SUDO zypper addrepo -f http://mirrors.163.com/openSUSE/distribution/leap/15.1/repo/non-oss dis-repo-non-oss163
  $SUDO zypper refresh
  $SUDO zypper install gcc gcc-c++
  $SUDO zypper install unzip libclang7 clang7-devel
}

install_clang() {
  if [ "$OS_NAME" == "Ubuntu" ]; then
    $SUDO apt-get install wget unzip -y
    install_clang_on_ubuntu
  elif [[ $OS_NAME == "CentOS"* ]] && [[ $OS_VERSION == "7"* ]]; then
    $SUDO yum install wget unzip -y
    install_clang_on_centos
  elif [[ "$OS_NAME" == "SLES"* ]] && [[ $OS_VERSION == "12"* ]]; then
    install_clang_on_sles
  else
    echo "WARNING: uncertified os type:version $OS_NAME:$OS_VERSION. Ait transplt installation may be incorrect!!!"
    # try to install clang
    $SUDO apt-get install wget unzip libclang-14-dev clang-14 -y
  fi
}

# Download and unzip config.zip, headers.zip
download_config_and_headers() {
  cwd=$(pwd)

  ori_mask=$(umask)
  umask 022
  if [ "$AIT_DOWNLOAD_PATH" != "" ]; then
    DOWNLOAD_PATH=$AIT_DOWNLOAD_PATH
  else
    DOWNLOAD_PATH=$(python3 -c "import app_analyze; print(app_analyze.__path__[0])")
  fi

  if [ $? -ne 0  ]; then
      echo "Downloading failed"
      return
  fi

  if [ "$AIT_INSTALL_FIND_LINKS" != "" ]; then
      cp "$AIT_INSTALL_FIND_LINKS/config" ./ -r
      cp "$AIT_INSTALL_FIND_LINKS/headers" ./ -r
  else 
    cd $DOWNLOAD_PATH \
      && wget -O config.zip https://ait-resources.obs.cn-south-1.myhuaweicloud.com/config.zip \
      && unzip -o -q config.zip \
      && rm config.zip -f \
      && wget -O headers.zip https://ait-resources.obs.cn-south-1.myhuaweicloud.com/headers.zip \
      && unzip -o -q headers.zip \
      && rm headers.zip -f
  fi

  umask $ori_mask
  cd $cwd
}

download_config_and_headers

if [ $# -gt 0 ] && [ "$1" == '--full' ]; then
  # Install clang
  install_clang

  $SHELL
fi
