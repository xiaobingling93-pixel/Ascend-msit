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


CUR_PATH=$(dirname $(readlink -f $0))

# copy source code to tests, and test
function copy_source_code_dir_to_tests() {
    cp -rf $CUR_PATH/../app_analyze $CUR_PATH/
}

function del_source_code_from_tests() {
    rm -rf $CUR_PATH/app_analyze
}

# download files from obs
OBS_SOURCE=https://ait-resources.obs.cn-south-1.myhuaweicloud.com

function download_from_obs() {
    for i in "$@"; do
        echo "downloading $i"
        wget -O $CUR_PATH/app_analyze/$i $OBS_SOURCE/$i > /dev/null 2>&1
        unzip -o $CUR_PATH/app_analyze/$i -d $CUR_PATH/app_analyze/ > /dev/null
    done
}

function install_requirements() {
    pip3 install -r ../requirements.txt
}

install_requirements

copy_source_code_dir_to_tests

download_from_obs config.zip headers.zip

if [ $? != 0 ]; then
    echo "download from obs failed"
    del_source_code_from_tests
    exit 1
fi

if [ -z "$PYTHONPATH" ]; then
    PYTHONPATH="$CUR_PATH/app_analyze/:$PYTHONPATH"
else
    PYTHONPATH="$CUR_PATH/app_analyze/"
fi

echo "PYTHONPATH: $PYTHONPATH"

chmod -R 750 $CUR_PATH/resources

coverage run -m pytest $CUR_PATH --disable-warnings
if [ $? != 0 ]; then
    echo "coverage run failed! "
    del_source_code_from_tests
    exit 1
fi

coverage combine $CUR_PATH
coverage report -m --omit="test_*.py" -i > $CUR_PATH/test.coverage

coverage_line=$(awk '/TOTAL/{print $4}' $CUR_PATH/test.coverage | cut -d '%' -f 1)
echo "coverage_line=$coverage_line%"

target=50  # Current is only 51%
if [[ "$coverage_line" -ne "" && "$coverage_line" -lt "$target" ]]; then
    echo "coverage failed! coverage_line=$coverage_line%, Coverage does not achieve target(${target}%), Please add ut case."
    del_source_code_from_tests
    exit 1
fi

del_source_code_from_tests
del_source_code_from_tests
exit 0
