# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
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

import torch.nn as nn


class SimpleModel_1(nn.Module):
    def __init__(self):
        super(SimpleModel_1, self).__init__()
        self.linear = nn.Linear(10, 20)
        self.dropout = nn.Dropout(0.5)


class MLPModel(nn.Module):
    def __init__(self):
        super(MLPModel, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 30)
        )


class AttentionModel(nn.Module):
    def __init__(self):
        super(AttentionModel, self).__init__()
        self.attention = nn.MultiheadAttention(10, 1)


class SimpleModel_2(nn.Module):
    def __init__(self, with_rope):
        super(SimpleModel_2, self).__init__()
        self.embed = nn.Embedding(10, 10)
        self.linear = nn.Linear(10, 10)
        self.dropout = nn.Dropout(0.5)
        self.config = type('Config', (object,), {'model_type': 'qwen_model'})


class SimpleModel_3(nn.Module):
    def __init__(self):
        super(SimpleModel_3, self).__init__()
        self.layer1 = nn.Linear(10, 20)
        self.layer2 = nn.Sequential(
            nn.Linear(20, 30),
            nn.Dropout(),
            nn.Linear(30, 40)
        )
        self.attention = nn.MultiheadAttention(60, 6)
        self.norm1 = nn.LayerNorm(60)
        self.norm2 = nn.LayerNorm(60)


class Config:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class EmptyModule(nn.Module):
    def __init__(self):
        super(EmptyModule, self).__init__()

    def forward(self, x):
        return x


class NoConfigModule(nn.Module):
    def __init__(self):
        super(NoConfigModule, self).__init__()
        self.linear = nn.Linear(10, 20)

    def forward(self, x):
        return self.linear(x)