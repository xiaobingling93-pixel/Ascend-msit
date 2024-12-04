# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from __future__ import print_function

import os
import logging

import numpy as np

from modelslim.pytorch.weight_compression import CompressConfig, Compressor

# 定义一个函数用于创建目录，如果目录不存在则创建
def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, mode=0o750)  # 创建目录并设置权限
    return path

# 定义主函数，用于执行权重压缩流程
def main(root):
    # 构建待压缩权重文件的路径
    weight_path = os.path.join(root, "quant_weight.npy")
    # 构建保存压缩结果的根目录路径
    save_path = f"{os.environ['PROJECT_PATH']}/output/weight_compression"
    # 创建保存压缩索引的目录
    index_root = make_dir(os.path.join(save_path, 'index'))
    # 创建保存压缩权重的目录
    weight_root = make_dir(os.path.join(save_path, 'weight'))
    # 创建保存压缩信息的目录
    info_root = make_dir(os.path.join(save_path, 'info'))

    # 配置压缩参数
    config = CompressConfig(
        do_pseudo_sparse=False,  # 是否使用伪稀疏压缩
        sparse_ratio=1,  # 稀疏比率
        is_debug=True,  # 是否开启调试模式
        record_detail_root=save_path,  # 记录压缩详情的根目录
        multiprocess_num=2  # 多进程数量
    )
    # 创建压缩器实例
    compressor = Compressor(config, weight_path)
    # 执行压缩操作
    compress_weight, compress_index, compress_info = compressor.run()

    # 导出压缩后的权重文件
    compressor.export(compress_weight, weight_root)
    # 导出压缩后的索引文件
    compressor.export(compress_index, index_root)
    # 导出压缩信息，指定数据类型为int64
    compressor.export(compress_info, info_root, dtype=np.int64)

# 从环境变量中获取资源目录的根路径
src_root = f"{os.environ['PROJECT_PATH']}/resource/weight_compression"
# 调用主函数执行权重压缩
main(src_root)
# 打印权重压缩成功的信息
logging.info("Weight Compression success!")