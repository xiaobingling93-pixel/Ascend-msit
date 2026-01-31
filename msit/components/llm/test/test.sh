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

CUR_PATH=$(dirname $(readlink -f $0))
COMPONENTS_PATH=`python -c 'import components; print(components.__path__[0])'`
SOURCE_CODE_PATH=$COMPONENTS_PATH/../msit_llm
echo "CUR_PATH=$CUR_PATH, COMPONENTS_PATH=$COMPONENTS_PATH, SOURCE_CODE_PATH=$SOURCE_CODE_PATH"

if [ -f "../requirements.txt" ]; then
    pip3 install -r ../requirements.txt
fi

if [ -f "$CUR_PATH/resources" ]; then
    chmod -R 750 $CUR_PATH/resources
fi

coverage run --source $SOURCE_CODE_PATH -m pytest -vv $CUR_PATH/testcase --disable-warnings

RETURN_CODE=0
if [ $? == 0 ]; then
    coverage combine $CUR_PATH
    coverage report -m --omit="test_*.py" -i > $CUR_PATH/test.coverage
    coverage_rate=$(awk '/TOTAL/{print $4}' $CUR_PATH/test.coverage | cut -d '%' -f 1)
    echo "coverage_rate=$coverage_rate%"

    coverage_target=50  # Current is only 51%
    if [[ "$coverage_rate" -ne "" && "$coverage_rate" -lt "$target" ]]; then
        echo "coverage rate too low(<${coverage_target}%), currently reaches only ${coverage_rate}%."
        RETURN_CODE=1
    fi
else
    echo "coverage run failed! "
    RETURN_CODE=1
fi

exit $RETURN_CODE
