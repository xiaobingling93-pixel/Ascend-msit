import argparse
from pathlib import Path

from modelevalstate.config.config import  DeployPolicy

import  modelevalstate.optimizer.optimizer as optimizer
import modelevalstate.train.source_to_train as train

def main():
    parser = argparse.ArgumentParser(description="msserviceprofiler command line tool")

    # 创建子命令解析器
    subparsers = parser.add_subparsers(dest="command", help="sub-command help")

    # 创建 train 子命令解析器
    parser_train = subparsers.add_parser("train", help="train help")
    parser_train.add_argument("-i", "--input", default=None, type=Path, required=True)
    parser_train.add_argument("-o", "--output", default=Path("output"), type=Path)
    # 创建 optimizer 子命令解析器
    parser_optimizer = subparsers.add_parser("optimizer", help="optimizer help")
    parser_optimizer.add_argument("-lb", "--load_breakpoint", default=False, action="store_true",
                        help="Continue from where the last optimization was aborted.")
    parser_optimizer.add_argument("-d", "--deploy_policy", default=DeployPolicy.single.value,
                        choices=[k.value for k in list(DeployPolicy)],
                        help="Indicates whether the multi-node running policy is used.")
    parser_optimizer.add_argument("--backup", default=False, action="store_true",
                        help="Whether to back up data.")
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