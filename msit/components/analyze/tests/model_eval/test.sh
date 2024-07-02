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

cur_dir=$(pwd)

# copy source code to tests, and test
function copy_source_code_dir_to_tests() {
    cp -rf ${cur_dir}/../../model_evaluation ${cur_dir}/
}

function del_source_code_from_tests() {
    rm -rf ${cur_dir}/model_evaluation
}

copy_source_code_dir_to_tests

coverage run -p -m unittest
if [ $? != 0 ]; then
    echo "coverage run failed! "
    del_source_code_from_tests
    exit 1
fi

coverage combine
coverage report -m --omit="test_*.py" > ${cur_dir}/test.coverage

coverage_line=`cat ${cur_dir}/test.coverage | grep "TOTAL" | awk '{print $4}' | awk '{print int($0)}'`

target=60
if [ ${coverage_line} -lt ${target} ]; then
    echo "coverage failed! coverage_line=${coverage_line}, Coverage does not achieve target(${target}%), Please add ut case."
    del_source_code_from_tests
    exit 1
fi

echo "coverage_line=${coverage_line}"

del_source_code_from_tests

exit 0
