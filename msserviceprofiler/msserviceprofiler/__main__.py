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

import msserviceprofiler.modelevalstate.optimizer.optimizer as optimizer
import msserviceprofiler.modelevalstate.train.source_to_train as train


def main():
    parser = argparse.ArgumentParser(description="msserviceprofiler command line tool")

    # 创建子命令解析器
    subparsers = parser.add_subparsers(dest="command", help="sub-command help")

    # 创建 train 子命令解析器
    parser_train = subparsers.add_parser("train", help="train help")
    parser_train.add_argument("-i", "--input", default=None, type=Path, required=True)
    parser_train.add_argument("-o", "--output", default=Path("output"), type=Path)
    parser_train.add_argument(
        "-t", 
        "--type", 
        type=str, 
        choices=["vllm", "mindie"], 
        default="mindie",
        help="Specify the type, either 'vllm' or 'mindie' (default: mindie)"
    )
    # 创建 optimizer 子命令解析器
    parser_optimizer = subparsers.add_parser("optimizer", help="optimizer help")
    parser_optimizer.add_argument("-lb", "--load_breakpoint", default=False, action="store_true",
                        help="Continue from where the last optimization was aborted.")
    parser_optimizer.add_argument("-d", "--deploy_policy", default=DeployPolicy.single.value,
                        choices=[k.value for k in list(DeployPolicy)],
                        help="Indicates whether the multi-node running policy is used.")
    parser_optimizer.add_argument("--backup", default=False, action="store_true",
                        help="Whether to back up data.")
    parser_optimizer.add_argument("-b", "--benchmark_policy", default=BenchMarkPolicy.benchmark.value,
                        choices=[k.value for k in list(BenchMarkPolicy)],
                        help="Whether to use custom performance indicators.")
    # 解析命令行参数
    args = parser.parse_args()

    # 根据子命令执行相应的操作
    if args.command == "train":
        train.main(args)
    elif args.command == "optimizer":
        optimizer.main(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()