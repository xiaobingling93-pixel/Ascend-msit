# Copyright (c) 2023-2023 Huawei Technologies Co., Ltd. All rights reserved.
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

#!/bin/bash

set -u

pwd_dir=$(dirname $(readlink -f "$0"))

# copy ait_llm to test file, and test
cp ${pwd_dir}/../llm ${pwd_dir}/ -rf

coverage run -m -p pytest testcase/*/test_*.py

ret=$?
if [ $ret != 0 ]; then
    echo "coverage run failed! "
    exit -1
fi

coverage combine
coverage report -m --omit="test_*.py" -i > ${pwd_dir}/test.coverage

coverage_line=`cat ${pwd_dir}/test.coverage | grep "TOTAL" | awk '{print $4}' | awk '{print int($0)}'`

echo "coverage_line=${coverage_line}"
