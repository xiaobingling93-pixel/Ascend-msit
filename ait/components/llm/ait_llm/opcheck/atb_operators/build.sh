#!/bin/bash
# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
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
set -e
SCRIPT_DIR=$(cd $(dirname $0); pwd)
AIT_LLM_INSTALL_PATH="$(python3 -c 'import ait_llm, os; print(os.path.dirname(os.path.abspath(ait_llm.__file__)))')"
IGNORE_INFO="If not using opcheck, ignore this error."

echo SCRIPT_DIR: $SCRIPT_DIR

function download_nlohmann_json()
{
    if [ -d "$SCRIPT_DIR/dependency/nlohmann" ]; then
        echo "[INFO] $SCRIPT_DIR/dependency/nlohmann already exists, skip downloading"
        return
    fi

    JSON_BASE_URL="https://github.com/nlohmann/json/archive/refs/tags"
    if [[ "$NLOHMAN_JSON_LINE" =~ "v3_11_1" ]]; then
        JSON_VERSION="3.11.1"
    elif [[ "$NLOHMAN_JSON_LINE" =~ "v3_11_2" ]]; then
        JSON_VERSION="3.11.2"
    elif [[ "$NLOHMAN_JSON_LINE" =~ "v3_11_3" ]]; then
        JSON_VERSION="3.11.3"
    else
        echo "[WARNING] Cannot find nlohmann json version, will treat as 3.11.2"
        JSON_VERSION="3.11.2"
    fi
    JSON_TAR="v${JSON_VERSION}.tar.gz"
    JSON_URL="$JSON_BASE_URL/$JSON_TAR"
    echo "JSON_URL=$JSON_URL"

    mkdir -p $SCRIPT_DIR/dependency
    cd $SCRIPT_DIR/dependency
    if [ "$AIT_INSTALL_FIND_LINKS" != "" ]; then 
        cp "$AIT_INSTALL_FIND_LINKS/$JSON_TAR" ./
    else
        wget --no-check-certificate -c $JSON_URL
    fi 
    
    if [ "$AIT_DOWNLOAD_PATH" != "" ]; then 
        mv $JSON_TAR "$AIT_DOWNLOAD_PATH"
        cd -
        exit 0
    fi
    tar xf $JSON_TAR

    JSON_FILE_NAME="json-$JSON_VERSION"
    if [ ! -d $JSON_FILE_NAME ] ; then
        echo "[ERROR] $JSON_FILE_NAME not exists. Check if anything wrong with downloading. $IGNORE_INFO"
        exit 1
    fi
    mv $JSON_FILE_NAME/include/nlohmann ./

    rm -rf $JSON_FILE_NAME
    cd -
}

if [ "$AIT_DOWNLOAD_PATH" != "" ]; then 
    download_nlohmann_json
    exit
fi

if [ "$AIT_LLM_INSTALL_PATH" == "" ]; then
    echo "[ERROR] ait_llm not found in python packages. Make sure ait_llm is installed for pip. $IGNORE_INFO"
    exit 1
fi

if [ "$ASCEND_TOOLKIT_HOME" == "" ]; then
    echo "[ERROR] ASCEND_TOOLKIT_HOME is empty. Make sure CANN toolkit is installed correctly. $IGNORE_INFO"
    exit 1
fi

if [ "$ATB_HOME_PATH" == "" ]; then
    echo "[ERROR] ATB_HOME_PATH is empty. Make sure atb is installed correctly. $IGNORE_INFO"
    exit 1
fi

if [ "$ATB_SPEED_HOME_PATH" == "" ]; then
    echo "[ERROR] ATB_SPEED_HOME_PATH is empty. Make sure mindie_atb_models is configured correctly. $IGNORE_INFO"
    exit 1
fi

if [ ! -e "$ATB_HOME_PATH/lib/libatb.so" ]; then
    echo "[ERROR] $ATB_HOME_PATH/lib/libatb.so not exists. Make sure atb is installed correctly. $IGNORE_INFO"
    exit 1
fi

NLOHMAN_JSON_LINE=`nm -D $ATB_HOME_PATH/lib/libatb.so | grep -i nlohmann | head -n 1`
if [ "$NLOHMAN_JSON_LINE" = "" ]; then
    echo "nlohmann json info not found in $ATB_HOME_PATH/lib/libatb.so. This shouldn't happen. make sure atb is installed correctly"
fi

if [[ "$NLOHMAN_JSON_LINE" =~ "cxx11" ]]; then
    CMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=1"
else
    CMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"
fi
echo "CMAKE_CXX_FLAGS=$CMAKE_CXX_FLAGS"


download_nlohmann_json

if [ -d "$SCRIPT_DIR/build" ]; then
    rm -rf $SCRIPT_DIR/build
fi

mkdir -p $SCRIPT_DIR/build

cd $SCRIPT_DIR/build
cmake .. -DCMAKE_INSTALL_PREFIX=$AIT_LLM_INSTALL_PATH/opcheck -DCMAKE_CXX_FLAGS=$CMAKE_CXX_FLAGS
make -j4 && make install
echo "[INFO] Copied $PWD/build/libopchecker.so -> $AIT_LLM_INSTALL_PATH/opcheck/libopchecker.so"
cd -
