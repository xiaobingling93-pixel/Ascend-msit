# -*- coding: utf-8 -*-
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
model = dict(
    type='cv',
    batch_size=1,
    engine=dict(
        dataset=dict(
            type='imagenet',
            dataset_path='./dataset/test_img/',
            label_path='./dataset/label.txt',
        ),
        pre_process=dict(
            type='classification',
            worker=1,
            resize=256,
            center_crop=[224, 224],
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            dtype='fp32',
        ),
        inference=dict(
            type='onnx',
            model='./onnx/squeezenet1_1.onnx',
        ),
        model_convert=dict(
            type='atc',
        ),
        post_process=dict(
            type='classification',
        ),
        evaluate=dict(type='classification', topk=[1, 5]),
    ),
)
