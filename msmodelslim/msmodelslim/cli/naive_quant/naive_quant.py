# Copyright Huawei Technologies Co., Ltd. 2025. All rights reserved.
from msmodelslim.app.naive_quantization.naive_entrance import NaiveEntrance


def main(args):
    cli = NaiveEntrance()
    cli.run_quantization(args)
