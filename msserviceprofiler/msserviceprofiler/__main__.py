# -*- coding: utf-8 -*-
# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
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
from pathlib import Path

from msserviceprofiler.modelevalstate.config.config import DeployPolicy, BenchMarkPolicy


_RUN_MODES = ["train", "optimizer", "advisor"]
RUN_MODES = namedtuple("RUN_MODES", _RUN_MODES)(*_RUN_MODES)
COMMON_ARGS = []

def run_train(args, **kwargs):
    from msserviceprofiler.modelevalstate.train import source_to_train
    source_to_train.main(args)


def sub_parser_train(subparsers):
    parser = subparsers.add_parser(
        RUN_MODES.train, formatter_class=argparse.ArgumentDefaultsHelpFormatter, help="train for auto optimize"
    )

    parser = subparsers.add_parser("train", help="train help")
    parser.add_argument("-i", "--input", default=None, type=Path, required=True)
    parser.add_argument("-o", "--output", default=Path("output"), type=Path)
    parser.add_argument(
        "-t", 
        "--type", 
        type=str, 
        choices=["vllm", "mindie"], 
        default="mindie",
        help="Specify the type, either 'vllm' or 'mindie' (default: mindie)"
    )
    for ii in COMMON_ARGS:
        parser.add_argument(*ii.get("args", []), **ii.get("kwargs", {}))
    parser.set_defaults(func=run_train)


def run_optimizer(args, **kwargs):
    from import msserviceprofiler.modelevalstate.optimizer import optimizer
    optimizer.main(args)


def sub_parser_optimizer(subparsers):
    parser = subparsers.add_parser(
        RUN_MODES.optimizer, formatter_class=argparse.ArgumentDefaultsHelpFormatter, help="optimize for performance"
    )

    parser.add_argument("-lb", "--load_breakpoint", action="store_true",
                        help="Continue from where the last optimization was aborted.")
    parser.add_argument("-d", "--deploy_policy", default=DeployPolicy.single.value,
                        choices=[k.value for k in list(DeployPolicy)],
                        help="Indicates whether the multi-node running policy is used.")
    parser.add_argument("--backup", default=False, action="store_true",
                        help="Whether to back up data.")
    parser.add_argument("-b", "--benchmark_policy", default=BenchMarkPolicy.benchmark.value,
                        choices=[k.value for k in list(BenchMarkPolicy)],
                        help="Whether to use custom performance indicators.")
    for ii in COMMON_ARGS:
        parser.add_argument(*ii.get("args", []), **ii.get("kwargs", {}))
    parser.set_defaults(func=run_optimizer)


def run_advisor(args, **kwargs):
    from import msserviceprofiler.modelevalstate.optimizer import optimizer
    optimizer.main(args)


def sub_parser_advisor(subparsers):
    parser = subparsers.add_parser(
        RUN_MODES.advisor, formatter_class=argparse.ArgumentDefaultsHelpFormatter, help="advisor for performance"
    )

    parser.add_argument("-lb", "--load_breakpoint", action="store_true",
                        help="Continue from where the last optimization was aborted.")
    parser.add_argument("-d", "--deploy_policy", default=DeployPolicy.single.value,
                        choices=[k.value for k in list(DeployPolicy)],
                        help="Indicates whether the multi-node running policy is used.")
    parser.add_argument("--backup", default=False, action="store_true",
                        help="Whether to back up data.")
    parser.add_argument("-b", "--benchmark_policy", default=BenchMarkPolicy.benchmark.value,
                        choices=[k.value for k in list(BenchMarkPolicy)],
                        help="Whether to use custom performance indicators.")
    for ii in COMMON_ARGS:
        parser.add_argument(*ii.get("args", []), **ii.get("kwargs", {}))
    parser.set_defaults(func=run_optimizer)

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="msserviceprofiler command line tool")
    subparsers = parser.add_subparsers(help="sub-command help")
    sub_parser_train(subparsers)
    sub_parser_optimizer(subparsers)
    args, _ = parser.parse_known_args()

    # run
    if hasattr(args, "func"):
        args.func(args=args, **vars(args))
    else:
        parser.print_help()

if __name__ == "__main__":
    main()