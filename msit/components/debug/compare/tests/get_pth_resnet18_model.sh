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
SOC_VERSION=""

try_download_url() {
    local _url=$1
    local _packet=$2
    cmd="wget $_url --no-check-certificate -O $_packet"
    $cmd #>/dev/null 2>&1
    ret=$?
    if [ "$ret" == 0 -a -s "$_packet" ]; then
        echo "download cmd:$cmd targetfile:$ OK"
    else
        echo "downlaod targetfile by $cmd Failed please check network or manual download to target file"
        return $ret_failed
    fi
}

function get_convert_file()
{
    local convert_url="https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/built-in/cv/Resnet18_for_PyTorch/resnet18_pth2onnx.py"
    wget $convert_url -O $1 --no-check-certificate
}

function convert_onnx_to_om()
{
    local _input_model=$1
    local _soc_version=$SOC_VERSION
    local _framework=5
    local _input_shape="image:1,3,224,224"

    local _output_model="$CUR_PATH/om/resnet18_static"
    local _output_path="$_output_model.om"
    if [ ! -f $_output_path ];then
        local _cmd="atc --model=$_input_model --output=$_output_model --framework=$_framework\
        --soc_version=$_soc_version --input_shape $_input_shape"
        $_cmd || { echo "atc run $_cmd failed"; return ret_failed; }
    fi
}

function get_npu_type()
{
    get_npu_310=`lspci | grep d100`
    get_npu_310P3=`lspci | grep d500`
    get_npu_310B=`lspci | grep d107`
    if [[ $get_npu_310 != "" ]];then
        SOC_VERSION="Ascend310"
        echo "npu is Ascend310"
    elif [[ $get_npu_310P3 != "" ]];then
        SOC_VERSION="Ascend310P3"
        echo "npu is Ascend310P3"
    elif [[ $get_npu_310B != "" ]];then
        SOC_VERSION="Ascend310B"
        echo "npu is Ascend310B"
    else
        return $ret_failed
    fi
}

main()
{
    get_npu_type || { echo "can't find supported npu";return $ret_failed; }
    pth_url="https://download.pytorch.org/models/resnet18-f37072fd.pth"
    pth_file="$CUR_PATH/onnx/pth_resnet18.pth"
    if [ ! -f $pth_file ]; then
        try_download_url $pth_url $pth_file || { echo "donwload stubs failed";return $ret_failed; }
    fi
    org_onnx_file="$CUR_PATH/onnx/pth_resnet18.onnx"
    if [ ! -f $org_onnx_file ]; then
        convert_file_path=$CUR_PATH/onnx/resnet18_pth2onnx.py
        get_convert_file $convert_file_path || { echo "get convert file failed";return $ret_failed; }
        chmod 750 $convert_file_path
        cd $CUR_PATH/onnx/
        python3 $convert_file_path --checkpoint $pth_file --save_dir $CUR_PATH/onnx/resnet18.onnx || { echo "convert pth to onnx failed";return $ret_failed; }
        mv $CUR_PATH/onnx/resnet18.onnx $org_onnx_file
        cd -
    fi
    static_onnx_file="$CUR_PATH/onnx/resnet18_static.onnx"
    if [ ! -f $static_onnx_file ]; then
        onnxsim $org_onnx_file $static_onnx_file || { echo "onnxsim failed!";return $ret_failed; }
    fi
    convert_onnx_to_om $static_onnx_file
}

main "$@"
exit $?