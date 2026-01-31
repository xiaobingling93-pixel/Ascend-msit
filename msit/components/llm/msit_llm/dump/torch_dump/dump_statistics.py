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
import os
import re
import csv

import torch
from components.utils.file_open_check import ms_open
from components.utils.util import safe_int


def dump_statistics(feat, feat_path: str, module_name, dump_type) -> None:
    #提取dump_type_name作为字符串
    #示例：<bound method Module.type of LayerNorm((4,), eps=1e-05, elementwise_affine=True)>，提取结果为：LayerNorm
    dump_type_name = str(dump_type).split('of ')[-1].split('(')[0]

    #获取输入类型和Index
    dump_data_name = os.path.basename(os.path.normpath(feat_path))
    input_output = "Input" if re.match(r"(input)(?:_(\d+))?", dump_data_name) else "Output"
    index = 0 
    if len(dump_data_name.split('_')) > 1:
        index = safe_int(dump_data_name.split('_')[-1])
    
    #定义csv文件名
    output_csv_path = os.path.join(os.path.dirname(feat_path), 'statistics.csv')

    #打开csv文件，用追加数据的方式进行写入
    with ms_open(output_csv_path, mode="a") as file:
        writer = csv.writer(file)

        file_is_empty = os.path.getsize(output_csv_path) == 0
        if file_is_empty:
            writer.writerow([
                'NodeName', 'NodeType', 'InputOutput', 'Index', 'DataType', 'Shape', 'Max', 'Min', 'Mean', 'L2Norm',
            ])

        data_tensor = feat.float()
        writer.writerow([
            module_name,     
            dump_type_name,
            input_output,
            index,
            feat.dtype,
            tuple(data_tensor.shape),
            torch.max(data_tensor).item(),
            torch.min(data_tensor).item(),
            torch.mean(data_tensor).item(),
            torch.norm(data_tensor).item(),
        ]) 