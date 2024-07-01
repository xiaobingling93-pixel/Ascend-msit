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

set -u

cur_dir=$(dirname $(readlink -f "$0"))

chmod -R 750 $cur_dir/test_resource
# copy source code to tests, and test
function copy_source_code_dir_to_tests() {
    cp -rf ${cur_dir}/../../msquickcmp ${cur_dir}/
}

function del_source_code_from_tests() {
    rm -rf ${cur_dir}/../msquickcmp
}

declare -i ret_val=0

main() {
    copy_source_code_dir_to_tests

    export PYTHON_COMMAND=${2:-"python3"}

    ${PYTHON_COMMAND} -m pytest . -s
    ret_val=$?

    del_source_code_from_tests

    return $ret_val
}

main "$@"
exit $?

