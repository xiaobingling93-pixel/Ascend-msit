#!/bin/bash
# Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
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
        pip3 wheel ./backend/ -v
        pip3 install ./aclruntime-*.whl --force-reinstall

        pip3 wheel ./ -v
        pip3 install ./ais_bench-*.whl --force-reinstall
    fi
}

main "$@"
exit $?