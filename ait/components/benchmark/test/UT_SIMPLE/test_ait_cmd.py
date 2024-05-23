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
from ais_bench.infer.benchmark_process import args_rules
from ais_bench.infer.args_adapter import BenchMarkArgsAdapter
from ais_bench.infer.main_cli import BenchmarkCommand

benchmark_command = BenchmarkCommand("benchmark","hekp")
data_path = os.getenv("AIT_BENCHMARK_DT_DATA_PATH")
if not data_path:
    current_dir = os.path.dirname(os.path.abspath(__file__))
else:
    current_dir = data_path
os.chmod(os.path.join(current_dir, "../json_for_arg_test.json"), 0o750)
base_cmd_dict = {
    "--om-model": os.path.join(current_dir, "../testdata/resnet50/model/pth_resnet50_bs4.om"),
    "--input": os.path.join(current_dir, "../testdata/resnet50/input/fake_dataset_bin_nor"),
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
    "--acl-json-path": os.path.join(current_dir, "../json_for_arg_test.json"),
    "--output-batchsize-axis": "1",
    "--run-mode": "array",
    "--display-all-summary": "0",
    "--warmup-count": "1",
    "--dym-shape-range": "1~3,3,224,224-226",
    "--aipp-config": os.path.join(current_dir, "../aipp_config_files/actual_aipp_cfg.config"),
    "--energy-consumption": "0",
    "--npu-id": "0",
    "--backend": "trtexec",
    "--perf": "0",
    "--pipeline": "0",
    "--profiler-rename": "0",
    "--dump-npy": "0",
    "--divide-input": "0",
    "--threads": "1"
}

simple_cmd_dict = {
    "-om": os.path.join(current_dir, "../testdata/resnet50/model/pth_resnet50_bs4.om"),
    "-i": os.path.join(current_dir, "../testdata/resnet50/input/fake_dataset_bin_nor"),
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
    "-acl": os.path.join(current_dir, "../json_for_arg_test.json"),
    "-oba": "1",
    "-rm": "array",
    "-das": "0",
    "-wcount": "1",
    "-dr": "1~3,3,224,224-226",
    "-aipp": os.path.join(current_dir, "../aipp_config_files/actual_aipp_cfg.config"),
    "-ec": "0",
    "--npu-id": "0",
    "--backend": "trtexec",
    "--perf": "0",
    "--pipeline": "0",
    "--profiler-rename": "0",
    "--dump-npy": "0",
    "--divide-input": "0",
    "--threads": "1"
}


def cmd_dict_to_list(cmd_dict):
    cmd_list = ['test_ait_cmd.py']
    for key, value in cmd_dict.items():
        cmd_list.append(key)
        cmd_list.append(value)
    return cmd_list


def create_adapter(args):
    args_adapter = BenchMarkArgsAdapter (
            model=args.om_model,
            input_path=args.input,
            output=args.output,
            output_dirname=args.output_dirname,
            outfmt=args.outfmt,
            loop=args.loop,
            debug=args.debug,
            device=args.device,
            dym_batch=args.dym_batch,
            dym_hw=args.dym_hw,
            dym_dims=args.dym_dims,
            dym_shape=args.dym_shape,
            output_size=args.output_size,
            auto_set_dymshape_mode=args.auto_set_dymshape_mode,
            auto_set_dymdims_mode=args.auto_set_dymdims_mode,
            batchsize=args.batch_size,
            pure_data_type=args.pure_data_type,
            profiler=args.profiler,
            dump=args.dump,
            acl_json_path=args.acl_json_path,
            output_batchsize_axis=args.output_batchsize_axis,
            run_mode=args.run_mode,
            display_all_summary=args.display_all_summary,
            warmup_count=args.warmup_count,
            dym_shape_range=args.dym_shape_range,
            aipp_config=args.aipp_config,
            energy_consumption=args.energy_consumption,
            npu_id=args.npu_id,
            backend=args.backend,
            perf=args.perf,
            pipeline=args.pipeline,
            profiler_rename=args.profiler_rename,
            dump_npy=args.dump_npy,
            divide_input = args.divide_input,
            threads = args.threads
    )
    return args_adapter


@pytest.fixture
def cmdline_legal_args_full(monkeypatch):
    cmd_dict = base_cmd_dict
    case_cmd_list = cmd_dict_to_list(cmd_dict)
    monkeypatch.setattr('sys.argv', case_cmd_list)


def test_check_all_full_args_legality(cmdline_legal_args_full):
    """
        正确的命令，使用可选命令全称
    """
    parser = argparse.ArgumentParser()
    benchmark_command.add_arguments(parser)
    args = parser.parse_args()
    args = create_adapter(args)
    args = args_rules(args)
    assert args.input == os.path.join(current_dir, "../testdata/resnet50/input/fake_dataset_bin_nor")


@pytest.fixture
def cmdline_legal_args_simple(monkeypatch):
    cmd_dict = simple_cmd_dict
    case_cmd_list = cmd_dict_to_list(cmd_dict)
    monkeypatch.setattr('sys.argv', case_cmd_list)


def test_check_all_simple_args_legality(cmdline_legal_args_simple):
    """
        正确的命令，使用可选命令简称
    """
    parser = argparse.ArgumentParser()
    benchmark_command.add_arguments(parser)
    args = parser.parse_args()
    args = create_adapter(args)
    args = args_rules(args)
    assert args.input == os.path.join(current_dir, "../testdata/resnet50/input/fake_dataset_bin_nor")


@pytest.fixture
def cmdline_args_full_model_path(monkeypatch):
    cmd_dict = base_cmd_dict
    cmd_dict["--model"] = os.path.join(current_dir, "../testdata/resnet50/model/pth_ret50_bs4.om")
    case_cmd_list = cmd_dict_to_list(cmd_dict)
    monkeypatch.setattr('sys.argv', case_cmd_list)


def test_invalid_model_path(cmdline_args_full_model_path):
    """
        模型路径不存在
    """
    parser = argparse.ArgumentParser()
    benchmark_command.add_arguments(parser)
    with pytest.raises(SystemExit) as e:
        args = parser.parse_args()


@pytest.fixture
def cmdline_args_full_loop(monkeypatch):
    cmd_dict = base_cmd_dict
    cmd_dict["--loop"] = "-3"
    case_cmd_list = cmd_dict_to_list(cmd_dict)
    monkeypatch.setattr('sys.argv', case_cmd_list)


def test_loop_is_not_positive(cmdline_args_full_loop):
    """
        --loop为负数
    """
    parser = argparse.ArgumentParser()
    benchmark_command.add_arguments(parser)
    with pytest.raises(SystemExit) as e:
        args = parser.parse_args()


@pytest.fixture
def cmdline_args_full_batchsize(monkeypatch):
    cmd_dict = base_cmd_dict
    cmd_dict["--batch-size"] = "-3"
    case_cmd_list = cmd_dict_to_list(cmd_dict)
    monkeypatch.setattr('sys.argv', case_cmd_list)


def test_batchsize_is_not_positive(cmdline_args_full_batchsize):
    """
        --batchsize为负数
    """
    parser = argparse.ArgumentParser()
    benchmark_command.add_arguments(parser)
    with pytest.raises(SystemExit) as e:
        args = parser.parse_args()


@pytest.fixture
def cmdline_args_full_warmup(monkeypatch):
    cmd_dict = base_cmd_dict
    cmd_dict["--warmup-count"] = "-3"
    case_cmd_list = cmd_dict_to_list(cmd_dict)
    monkeypatch.setattr('sys.argv', case_cmd_list)


def test_warmup_count_is_not_positive(cmdline_args_full_warmup):
    """
        --warmup_count为负数
    """
    parser = argparse.ArgumentParser()
    benchmark_command.add_arguments(parser)
    with pytest.raises(SystemExit) as e:
        args = parser.parse_args()


@pytest.fixture
def cmdline_args_full_bsaxis(monkeypatch):
    cmd_dict = base_cmd_dict
    cmd_dict["--output-batchsize-axis"] = "-3"
    case_cmd_list = cmd_dict_to_list(cmd_dict)
    monkeypatch.setattr('sys.argv', case_cmd_list)


def test_output_batchsize_axis_is_not_positive(cmdline_args_full_bsaxis):
    """
        --output_batchsize_axis为负数
    """
    parser = argparse.ArgumentParser()
    benchmark_command.add_arguments(parser)
    with pytest.raises(SystemExit) as e:
        args = parser.parse_args()


@pytest.fixture
def cmdline_args_full_device(monkeypatch):
    cmd_dict = base_cmd_dict
    cmd_dict["--device"] = "1,234,257"
    case_cmd_list = cmd_dict_to_list(cmd_dict)
    monkeypatch.setattr('sys.argv', case_cmd_list)


def test_device_id_out_of_range(cmdline_args_full_device):
    """
        --device超出范围
    """
    parser = argparse.ArgumentParser()
    benchmark_command.add_arguments(parser)
    with pytest.raises(SystemExit) as e:
        args = parser.parse_args()


@pytest.fixture
def cmdline_args_full_outfmt(monkeypatch):
    cmd_dict = base_cmd_dict
    cmd_dict["--outfmt"] = "JSON"
    case_cmd_list = cmd_dict_to_list(cmd_dict)
    monkeypatch.setattr('sys.argv', case_cmd_list)


def test_illegal_outfmt(cmdline_args_full_outfmt):
    """
        --outfmt非法
    """
    parser = argparse.ArgumentParser()
    benchmark_command.add_arguments(parser)
    with pytest.raises(SystemExit) as e:
        args = parser.parse_args()