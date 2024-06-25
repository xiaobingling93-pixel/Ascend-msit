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

import os
import argparse

import pytest
from msit_benchmark.__main__ import get_cmd_instance


CUR_DIR = f"{os.path.dirname(__file__)}/"
PREFIX="benchmark_test_args_fake_"
FAKE_OM_PATH=PREFIX + "model.om"
FAKE_BIN_PATH=PREFIX + "data.bin"
FAKE_ACL_JSON_PATH=PREFIX + "acl.json"
FAKE_AIPP_CFG_PATH=PREFIX + "aipp.cfg"
INVALID_ARG = "--invalid_arg"

FULL_CMD_DICT = {
    "--om-model": FAKE_OM_PATH,
    "--input": FAKE_BIN_PATH,
    "--output": "output/",
    "--output-dirname": "outdir/",
    "--outfmt": "NPY",
    "--loop": "100",
    "--debug": "0",
    "--device": "0,1",
    "--dym-batch": "16",
    "--dym-hw": "224,224",
    "--dym-dims": "1,3,224,224",
    "--dym-shape": "1,3,224,224",
    "--output-size": "10000",
    "--auto-set-dymshape-mode": "0",
    "--auto-set-dymdims-mode": "0",
    "--batch-size": "16",
    "--pure-data-type": "zero",
    "--profiler": "0",
    "--dump": "0",
    "--acl-json-path": FAKE_ACL_JSON_PATH,
    "--output-batchsize-axis": "1",
    "--run-mode": "array",
    "--display-all-summary": "0",
    "--warmup-count": "1",
    "--dym-shape-range": "1~3,3,224,224-226",
    "--aipp-config": FAKE_AIPP_CFG_PATH,
    "--energy-consumption": "0",
    "--npu-id": "0",
    "--backend": "trtexec",
    "--perf": "0",
    "--pipeline": "0",
    "--profiler-rename": "0",
    "--dump-npy": "0",
    "--divide-input": "0",
    "--threads": "1",
}

SHORT_CMD_DICT = {
    "-om": FAKE_OM_PATH,
    "-i": FAKE_BIN_PATH,
    "-o": "output/",
    "-od": "outdir/",
    "--outfmt": "NPY",
    "--loop": "100",
    "--debug": "0",
    "-d": "0,1",
    "-db": "16",
    "-dhw": "224,224",
    "-dd": "1,3,224,224",
    "-ds": "1,3,224,224",
    "-outsize": "10000",
    "-asdsm": "0",
    "-asddm": "0",
    "--batch-size": "16",
    "-pdt": "zero",
    "-pf": "0",
    "--dump": "0",
    "-acl": FAKE_ACL_JSON_PATH,
    "-oba": "1",
    "-rm": "array",
    "-das": "0",
    "-wcount": "1",
    "-dr": "1~3,3,224,224-226",
    "-aipp": FAKE_AIPP_CFG_PATH,
    "-ec": "0",
    "--npu-id": "0",
    "--backend": "trtexec",
    "--perf": "0",
    "--pipeline": "0",
    "--profiler-rename": "0",
    "--dump-npy": "0",
    "--divide-input": "0",
    "--threads": "1",
}

def benchmark_argparse(argv):
    aa = get_cmd_instance()
    parser = argparse.ArgumentParser()
    aa.add_arguments(parser)
    args_parser = parser.parse_args(argv)
    return args_parser


def call_benchmark_cmd(argv):
    aa = get_cmd_instance()
    args = benchmark_argparse(argv)
    return aa.handle(args)


def cmd_dict_to_list(cmd_dict, new_args={}):
    cmd_list = []
    for key, value in cmd_dict.items():
        cmd_list.append(key)
        cmd_list.append(new_args.get(key, value))
    return cmd_list

def test_benchmark_argparse_given_valid_when_full_then_pass():
    args = benchmark_argparse(cmd_dict_to_list(FULL_CMD_DICT))
    assert args.om_model == FAKE_OM_PATH
    assert args.input == FAKE_BIN_PATH

def test_benchmark_argparse_given_valid_when_short_then_pass():
    args = benchmark_argparse(cmd_dict_to_list(SHORT_CMD_DICT))
    assert args.om_model == FAKE_OM_PATH
    assert args.input == FAKE_BIN_PATH


def test_benchmark_argparse_given_invalid_arg_when_full_then_error():
    new_args = {INVALID_ARG: "anything"}
    with pytest.raises(SystemExit) as e:
        args = benchmark_argparse(cmd_dict_to_list(FULL_CMD_DICT, new_args=new_args))
