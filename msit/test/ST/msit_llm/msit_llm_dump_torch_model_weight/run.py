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
import os.path

from main_torch import check_min_version, override_topp_and_topk, parse_args, get_model, precision, performance, \
    cli_demo, webUI


def main():
    # Will error if the minimal version of Transformers is not installed. Remove at your own risks.
    check_min_version('4.30.2')
    override_topp_and_topk()
    args = parse_args()
    tokenizer, model = get_model(args)
    from msit_llm import DumpConfig, register_hook

    dump_config = DumpConfig(dump_path=os.path.realpath("./ait_dump_torch_weight"),
                            dump_weight=True, mode=["module", "api"],
                            layer_name="root.transformer.encoder.layers.0*")
    register_hook(model, dump_config)

    if 'precision' in args.mode:
        precision(args, tokenizer, model)
    elif 'performance' in args.mode:
        performance(args, tokenizer, model)
    elif 'cli_demo' in args.mode:
        cli_demo(args, tokenizer, model)
    else:
        webUI(args, tokenizer, model)


if __name__ == '__main__':
    main()