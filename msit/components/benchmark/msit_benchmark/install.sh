#!/bin/bash
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

declare -i ret_ok=0
declare -i ret_run_failed=1

WHL_BASE_URL=$(python3 -c "from components.utils.install import get_public_url; print(get_public_url('whl_base_url'))")
ACLRUNTIME_SHA_BASE_URL=$(python3 -c "from components.utils.install import get_public_url; print(get_public_url('aclruntime_sha_base_url'))")
TOOLS_BAS_URL_SUFFIX=$(python3 -c "from components.utils.install import get_public_url; print(get_public_url('tools_base_url'))")
AIS_BENCH_SHA_BASE_URL=$(python3 -c "from components.utils.install import get_public_url; print(get_public_url('ais_bench_sha_base_url'))")
TOOLS_BAS_URL="git+$TOOLS_BAS_URL_SUFFIX"

download_and_install_aclruntime() {
    ACLRUNTIME_VERSION=`pip3 show aclruntime | awk '/Version: /{print $2}'`

    if [ "$arg_force_reinstall" = "--force-reinstall" ]; then
        echo "Force reinstall aclruntime"
    elif [ "$ACLRUNTIME_VERSION" = "0.0.2" ]; then
        echo "aclruntime==0.0.2 already installed, skip"
        return
    fi

    echo "download and install aclruntime"
    PYTHON3_MINI_VERSION=`python3 --version | cut -d'.' -f 2`
    PYTHON3_MINI_VERSION=`python3 --version | cut -d'.' -f 2`
    if [ "$PYTHON3_MINI_VERSION" = "7" ]; then
        SUB_SUFFIX="m"
    else
        SUB_SUFFIX=""
    fi
    echo "PYTHON3_MINI_VERSION=$PYTHON3_MINI_VERSION, SUB_SUFFIX=$SUB_SUFFIX"
    PLATFORM=$(uname -m)

    WHL_NAME="aclruntime-0.0.2-cp3${PYTHON3_MINI_VERSION}-cp3${PYTHON3_MINI_VERSION}${SUB_SUFFIX}-linux_${PLATFORM}.whl"
    echo "WHL_NAME=$WHL_NAME, URL=${WHL_BASE_URL}${WHL_NAME}"
    if [ "$NO_CHECK_CERTIFICATE" == "true" ]; then
        echo "[WARNING] --no-check will skip checking the certificate of the target website, posing security risk."
        wget --no-check-certificate -c "${WHL_BASE_URL}${WHL_NAME}"
    else
        wget -c "${WHL_BASE_URL}${WHL_NAME}"
    fi

    if [ $PYTHON3_MINI_VERSION -gt 11 ]; then
        echo "Unsupported python3 version"
        exit 1
    fi

    SHA_NAME="${WHL_NAME%.whl}.sha256"
    SHA_URL=${ACLRUNTIME_SHA_BASE_URL}${SHA_NAME}
    echo "Downloading aclruntime SHA from: ${SHA_URL}"
    if [ "$NO_CHECK_CERTIFICATE" == "true" ]; then
        sha256Value=$(curl -kfsSL "${SHA_URL}" | awk '{print $1}')
    else
        sha256Value=$(curl -fsSL "${SHA_URL}" | awk '{print $1}')
    fi

    sha256Data=$(sha256sum "$WHL_NAME" | cut -d' ' -f1)
    if [[ "${sha256Data}" != "${sha256Value}" ]]; then
        echo "Failed to verify sha256: $WHL_NAME"
        exit 1
    else
        echo "sha256 verification passed: $WHL_NAME"
    fi

    pip3 install $WHL_NAME $arg_force_reinstall && rm -f $WHL_NAME
    if [ $? -ne 0 ]; then
        echo "Downloading or installing from whl failed, will install from source code"
        pip3 install -v "${TOOLS_BAS_URL}#egg=aclruntime&subdirectory=ais-bench_workload/tool/ais_bench/backend" --force-reinstall
    fi
}

download_and_install_ais_bench() {
    AIS_BENCH_VERSION=`pip3 show ais_bench | awk '/Version: /{print $2}'`

    if [ "$arg_force_reinstall" = "--force-reinstall" ]; then
        echo "Force reinstall ais_bench"
    elif [ "$AIS_BENCH_VERSION" = "0.0.2" ]; then
        echo "ais_bench==0.0.2 already installed, skip"
        return
    fi

    WHL_NAME="ais_bench-0.0.2-py3-none-any.whl"
    echo "WHL_NAME=$WHL_NAME, URL=${WHL_BASE_URL}${WHL_NAME}"
    if [ "$NO_CHECK_CERTIFICATE" == "true" ]; then
        echo "[WARNING] --no-check will skip checking the certificate of the target website, posing security risk."
        wget --no-check-certificate -c "${WHL_BASE_URL}${WHL_NAME}"
    else
        wget -c "${WHL_BASE_URL}${WHL_NAME}"
    fi

    SHA_NAME="${WHL_NAME%.whl}.sha256"
    SHA_URL=${AIS_BENCH_SHA_BASE_URL}${SHA_NAME}
    echo "Downloading ais_bench SHA from: ${SHA_URL}"
    if [ "$NO_CHECK_CERTIFICATE" == "true" ]; then
        sha256Value=$(curl -kfsSL "${SHA_URL}" | awk '{print $1}')
    else
        sha256Value=$(curl -fsSL "${SHA_URL}" | awk '{print $1}')
    fi

    sha256Data=$(sha256sum "$WHL_NAME" | cut -d' ' -f1)
    if [[ "${sha256Data}" != "${sha256Value}" ]]; then
        echo "Failed to verify sha256: $WHL_NAME"
        exit 1
    else
        echo "sha256 verification passed: $WHL_NAME"
    fi

    pip3 install $WHL_NAME $arg_force_reinstall && rm -f $WHL_NAME
    if [ $? -ne 0 ]; then
        echo "Downloading or installing from whl failed, will install from source code"
        pip3 install -v "${TOOLS_BAS_URL}#egg=aclruntime&subdirectory=ais-bench_workload/tool/ais_bench/backend" --force-reinstall
    fi
}

ret=0
download_and_install_aclruntime
ret=$(( $ret + $? ))

download_and_install_ais_bench
ret=$(( $ret + $? ))

exit $ret