#!/bin/bash
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

declare -i ret_ok=0
declare -i ret_run_failed=1

check_python_package_is_install()
{
    local PYTHON_COMMAND=$1
    ${PYTHON_COMMAND} -c "import $2" >> /dev/null 2>&1
    ret=$?
    if [ $ret != 0 ]; then
        echo "python package:$2 not install"
        return 1
    fi
    return 0
}

check_env_valid()
{
    check_python_package_is_install ${PYTHON_COMMAND} "aclruntime" \
    || { echo "aclruntime package not install"; return $ret_run_failed;}

    check_python_package_is_install ${PYTHON_COMMAND} "ais_bench" \
    || { echo "ais_bench package not install"; return $ret_run_failed;}
}

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
    WHL_NAME="aclruntime-0.0.2-cp3${PYTHON3_MINI_VERSION}-cp3${PYTHON3_MINI_VERSION}${SUB_SUFFIX}-linux_$(uname -m).whl"
    BASE_URL="https://aisbench.obs.myhuaweicloud.com/packet/ais_bench_infer/0.0.2/ait/"
    echo "WHL_NAME=$WHL_NAME, URL=${BASE_URL}${WHL_NAME}"
    wget --no-check-certificate -c "${BASE_URL}${WHL_NAME}" && pip3 install $WHL_NAME --force-reinstall && rm -f $WHL_NAME
    if [ $? -ne 0 ]; then
        echo "Downloading or installing from whl failed, will install from source code"
        pip3 install -v 'git+https://gitee.com/ascend/tools.git#egg=aclruntime&subdirectory=ais-bench_workload/tool/ais_bench/backend' --force-reinstall
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
    BASE_URL="https://aisbench.obs.myhuaweicloud.com/packet/ais_bench_infer/0.0.2/ait/"
    echo "WHL_NAME=$WHL_NAME, URL=${BASE_URL}${WHL_NAME}"
    wget --no-check-certificate -c "${BASE_URL}${WHL_NAME}" && pip3 install $WHL_NAME --force-reinstall && rm -f $WHL_NAME
    if [ $? -ne 0 ]; then
        echo "Downloading or installing from whl failed, will install from source code"
        pip3 install -v 'git+https://gitee.com/ascend/tools.git#egg=aclruntime&subdirectory=ais-bench_workload/tool/ais_bench/backend' --force-reinstall
    fi
}

main()
{
      while [ -n "$1" ]
do
  case "$1" in
    -p|--python_command)
        PYTHON_COMMAND=$2
        shift
        ;;
    *)
        echo "$1 is not an option, please use --help"
        exit 1
        ;;
  esac
  shift
done

    [ "$PYTHON_COMMAND" != "" ] || { PYTHON_COMMAND="python3.7";echo "set default pythoncmd:$PYTHON_COMMAND"; }

    check_env_valid
    res=`echo $?`
    if [ $res = $ret_run_failed ]; then
        download_and_install_aclruntime
        download_and_install_ais_bench
    fi
}

main "$@"
exit $?