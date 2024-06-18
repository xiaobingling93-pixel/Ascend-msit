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

import argparse
import os
import re
from ais_bench.infer.benchmark_process import benchmark_process
from ais_bench.infer.args_adapter import BenchMarkArgsAdapter
from ais_bench.infer.args_check import (
    check_dym_string, check_dym_range_string, check_number_list, str2bool, check_positive_integer,
    check_batchsize_valid, check_nonnegative_integer, check_device_range_valid, check_om_path_legality,
    check_input_path_legality, check_output_path_legality, check_acl_json_path_legality,
    check_aipp_config_path_legality
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        "-m",
        type=check_om_path_legality,
        required=True,
        help="The path of the om model"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=check_input_path_legality,
        default=None,
        help="Input file or dir"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=check_output_path_legality,
        default=None,
        help="Inference data output path. The inference results are output to \
             the subdirectory named current date under given output path"
    )
    parser.add_argument(
        "--output_dirname",
        type=check_output_path_legality,
        default=None,
        help="Actual output directory name. \
             Used with parameter output, cannot be used alone. \
             The inference result is output to subdirectory named by output_dirname \
             under  output path. such as --output_dirname 'tmp', \
             the final inference results are output to the folder of  {$output}/tmp"
    )
    parser.add_argument(
        "--outfmt",
        default="BIN",
        choices=["NPY", "BIN", "TXT"],
        help="Output file format (NPY or BIN or TXT)"
    )
    parser.add_argument(
        "--loop",
        "-l",
        type=check_positive_integer,
        default=1,
        help="The round of the PureInfer."
    )
    parser.add_argument(
        "--debug",
        type=str2bool,
        default=False,
        help="Debug switch,print model information"
    )
    parser.add_argument(
        "--device",
        "-d",
        type=check_device_range_valid,
        default=0,
        help="The NPU device ID to use.valid value range is [0, 255]"
    )
    parser.add_argument(
        "--dymBatch",
        dest="dym_batch",
        type=check_positive_integer,
        default=0,
        help="Dynamic batch size param，such as --dymBatch 2"
    )
    parser.add_argument(
        "--dymHW",
        dest="dym_hw",
        type=check_dym_string,
        default=None,
        help="Dynamic image size param, such as --dymHW \"300,500\""
    )
    parser.add_argument(
        "--dymDims",
        dest="dym_dims",
        type=check_dym_string,
        default=None,
        help="Dynamic dims param, such as --dymDims \"data:1,600;img_info:1,600\""
    )
    parser.add_argument(
        "--dymShape",
        "--dym-shape",
        dest="dym_shape",
        type=check_dym_string,
        default=None,
        help="Dynamic shape param, such as --dymShape \"data:1,600;img_info:1,600\""
    )
    parser.add_argument(
        "--outputSize",
        dest="output_size",
        type=check_number_list,
        default=None,
        help="Output size for dynamic shape mode"
    )
    parser.add_argument(
        "--auto_set_dymshape_mode",
        type=str2bool,
        default=False,
        help="Auto_set_dymshape_mode"
    )
    parser.add_argument(
        "--auto_set_dymdims_mode",
        type=str2bool,
        default=False,
        help="Auto_set_dymdims_mode"
    )
    parser.add_argument(
        "--batchsize",
        type=check_batchsize_valid,
        default=None,
        help="Batch size of input tensor"
    )
    parser.add_argument(
        "--pure_data_type",
        type=str,
        default="zero",
        choices=["zero", "random"],
        help="Null data type for pure inference(zero or random)"
    )
    parser.add_argument(
        "--profiler",
        type=str2bool,
        default=False,
        help="Profiler switch"
    )
    parser.add_argument(
        "--dump",
        type=str2bool,
        default=False,
        help="Dump switch"
    )
    parser.add_argument(
        "--acl_json_path",
        type=check_acl_json_path_legality,
        default=None,
        help="Acl json path for profiling or dump"
    )
    parser.add_argument(
        "--output_batchsize_axis",
        type=check_nonnegative_integer,
        default=0,
        help="Splitting axis number when outputing tensor results, such as --output_batchsize_axis 1"
    )
    parser.add_argument(
        "--run_mode",
        type=str,
        default="array",
        choices=["array", "files", "tensor", "full"],
        help="Run mode"
    )
    parser.add_argument(
        "--display_all_summary",
        type=str2bool,
        default=False,
        help="Display all summary include h2d d2h info"
    )
    parser.add_argument(
        "--warmup_count",
        "--warmup-count",
        type=check_nonnegative_integer,
        default=1,
        help="Warmup count before inference"
        )
    parser.add_argument(
        "--dymShape_range",
        dest="dym_shape_range",
        type=check_dym_range_string,
        default=None,
        help="Dynamic shape range, such as --dymShape_range \"data:1,600~700;img_info:1,600-700\""
    )
    parser.add_argument(
        "--aipp_config",
        type=check_aipp_config_path_legality,
        default=None,
        help="File type: .config, to set actual aipp params before infer"
    )
    parser.add_argument(
        "--energy_consumption",
        type=str2bool,
        default=False,
        help="Obtain power consumption data for model inference"
    )
    parser.add_argument(
        "--npu_id",
        type=check_nonnegative_integer,
        default=0,
        help="The NPU ID to use.valid value range is [0, 255]"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=["trtexec"],
        help="Backend trtexec"
    )
    parser.add_argument(
        "--perf",
        type=str2bool,
        default=False,
        help="Perf switch"
    )
    parser.add_argument(
        "--pipeline",
        type=str2bool,
        default=False,
        help="Pipeline switch"
    )
    parser.add_argument(
        "--profiler_rename",
        type=str2bool,
        default=True,
        help="Profiler rename switch"
    )
    parser.add_argument(
        "--dump_npy",
        type=str2bool,
        default=False,
        help="dump data convert to npy"
    )
    parser.add_argument(
        "--divide_input",
        type=str2bool,
        default=False,
        help="Input datas need to be divided to match multi devices or not, \
            --device should be list"
    )
    parser.add_argument(
        '--threads',
        dest='threads',
        type=check_positive_integer,
        default=1,
        help="Number of threads for computing. \
            need to set --pipeline when setting threads number to be more than one."
    )
    benchmark_args = parser.parse_args()

    return benchmark_args


if __name__ == "__main__":
    args = get_args()

    args = BenchMarkArgsAdapter(args.model, args.input, args.output,
                args.output_dirname, args.outfmt, args.loop, args.debug, args.device,
                args.dym_batch, args.dym_hw, args.dym_dims, args.dym_shape, args.output_size,
                args.auto_set_dymshape_mode, args.auto_set_dymdims_mode, args.batchsize, args.pure_data_type,
                args.profiler, args.dump, args.acl_json_path, args.output_batchsize_axis, args.run_mode,
                args.display_all_summary, args.warmup_count, args.dym_shape_range, args.aipp_config,
                args.energy_consumption, args.npu_id, args.backend, args.perf, args.pipeline, args.profiler_rename,
                args.dump_npy, args.divide_input, args.threads)
    ret = benchmark_process(args)
    exit(ret)
