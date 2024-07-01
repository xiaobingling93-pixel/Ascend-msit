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
declare -i ret_failed=1
CUR_PATH=$(dirname $(readlink -f "$0"))
TEST_DATA_PATH=$CUR_PATH/../../benchmark/test/testdata/
OUTPUT_PATH=$CUR_PATH/output_datas

PYTHON_COMMAND="python3"


main() {
    if [ ! -d $OUTPUT_PATH ];then
        mkdir $OUTPUT_PATH || { echo "make output dir failed"; return $ret_failed; }
        chmod 750 $OUTPUT_PATH
    fi

    ${PYTHON_COMMAND} -m pytest -s $CUR_PATH/test_profile_cmd.py || {  echo "execute ST command failed!"; return $ret_failed; }
    return $ret_ok
}
main "$@"
exit $?