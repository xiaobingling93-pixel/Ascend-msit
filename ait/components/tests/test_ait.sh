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
declare -i ret_failed=1
CUR_PATH=$("pwd")
ALL_VALID_TEST_CASES=(/analyze/ /benchmark/ /convert/ /debug/compare/ /debug/surgeon/ /llm/ /profile/ /transplt/ /tensor_view/ /utils/)

function is_path_in_all_valid_test_cases() {
    for test_case in ${RUN_TESTCASES[@]}; do
        if [[ "$1" =~ "$test_case" ]]; then
            echo 1
            return
        fi
    done
    echo 0
}

function get_modified_module_list() {
    soft_link_path=/home/dcs-50/ait_test/ait/ait/components
    [[ -d $soft_link_path ]] || { echo "can't find origin dt data";return $ret_failed; }
    cur_testdata_path=$CUR_PATH/../benchmark/test/testdata
    [[ -d $cur_testdata_path ]] || { `ln -s $soft_link_path/benchmark/test/testdata $cur_testdata_path`; }
    modify_files=$CUR_PATH/../../../../modify_files.txt
    RUN_TESTCASES=()
    if [[ -f $modify_files ]];then
        echo "found modify_files"
        while read line
        do
            for test_case in ${ALL_VALID_TEST_CASES[@]}; do
                if [[ "$line" =~ "$test_case" ]]; then
                    RUN_TESTCASES=(${RUN_TESTCASES[@]} "$test_case")
                    echo "run $test_case DT"
                fi
            done
        done < $modify_files
    fi
}

main() {
    export dt_mode=${1:-"normal"} # or "pr"
    if [[ $dt_mode == "pr" ]];then
        get_modified_module_list
    else
        RUN_TESTCASES=${ALL_VALID_TEST_CASES[@]}
    fi
    echo "RUN_TESTCASES: ${RUN_TESTCASES[@]}"

    failed_case_names=""
    all_part_test_ok=0
    if [[ $PWD =~ "components/tests" ]]; then
        TEST_CASES=( $(find ../* -name test.sh) )  # In tests dir
    else
        TEST_CASES=( $(find ./* -name test.sh) )
    fi

    echo "pwd: $PWD, TEST_CASES: ${TEST_CASES[@]}"
    for test_case in ${TEST_CASES[@]}; do
        is_valid=$(is_path_in_all_valid_test_cases $test_case)
        echo ">>>> Current test_case=$test_case, is_valid=$is_valid"

        if [ $is_valid -eq 0 ]; then
            continue
        fi

        CASE_PATH=`dirname $test_case`
        cd $CASE_PATH
        bash test.sh
        cur_result=$?
        echo ">>>> test_case=$test_case, cur_result=$cur_result"
        if [ "$cur_result" -ne "0" ]; then
            failed_case_names="$failed_case_names, $test_case"
            all_part_test_ok=$(( $all_part_test_ok + $cur_result ))
        fi
        cd $CUR_PATH
    done

    echo "failed_case_names: ${failed_case_names:2}"  # Exclude the first ", "
    return $all_part_test_ok
}

main "$@"
exit $?