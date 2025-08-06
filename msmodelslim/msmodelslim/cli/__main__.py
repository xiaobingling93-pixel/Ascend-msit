# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import argparse
from pathlib import Path

from msmodelslim.app import QuantType, DeviceType
from msmodelslim.cli.naive_quantization.__main__ import main as quant_main

FAQ_HOME = "gitee repo: Ascend/msit/msmodelslim, wiki"
MIND_STUDIO_LOGO = "[Powered by MindStudio]"


def cmd_bool(cmd_arg):
    if cmd_arg == "True":
        return True
    elif cmd_arg == "False":
        return False
    raise argparse.ArgumentTypeError(f"{cmd_arg} should be True or False")


def cmd_path(cmd_arg):
    path = Path(cmd_arg)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"Invalid path: {path}. Path does not exist.")
    return path


def cmd_new_path(cmd_arg):
    path = Path(cmd_arg)
    if not path.parent.exists():
        raise argparse.ArgumentTypeError(f"Invalid path: {path}. Parent path does not exist.")
    return path


def main():
    parser = argparse.ArgumentParser(prog='msmodelslim',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=f"MsModelSlim(MindStudio Model-Quantization Tools), "
                                                 f"{MIND_STUDIO_LOGO}.\n"
                                                 "Providing functions such as model quantization and compression "
                                                 "based on Ascend.\n"
                                                 f"For any issue, refer FAQ first: {FAQ_HOME}")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Quant command
    quant_parser = subparsers.add_parser('quant', help='Model quantization')
    quant_parser.add_argument('--model_type', required=True,
                              help="Type of model to quantize (e.g. 'Qwen2.5-7B-Instruct', 'Qwen-QwQ-32B')")
    quant_parser.add_argument('--model_path', required=True, type=cmd_path,
                              help="Path to the original model")
    quant_parser.add_argument('--save_path', required=True, type=cmd_new_path,
                              help="Path to save quantized model")
    quant_parser.add_argument('--device', type=DeviceType, default=DeviceType.NPU, choices=DeviceType,
                              help="Target device type for quantization")
    quant_parser.add_argument('--config_path', type=cmd_path,
                              help="Explicit path to quantization config file")
    quant_parser.add_argument('--quant_type', type=QuantType, choices=QuantType,
                              help="Type of quantization to apply")
    quant_parser.add_argument('--trust_remote_code', type=cmd_bool, default=False,
                              help="Trust custom code (bool type, must be True or False). "
                                   "Please ensure the security of the loaded custom code file.")

    args = parser.parse_args()

    if args.command == 'quant':
        quant_main(args)
    else:
        # 可扩展其他组件
        parser.print_help()


if __name__ == '__main__':
    main()
