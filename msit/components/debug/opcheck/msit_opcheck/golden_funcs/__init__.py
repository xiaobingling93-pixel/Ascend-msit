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
from msit_opcheck.golden_funcs.logical_or import LogicalOrOperation
from msit_opcheck.golden_funcs.logical_and import LogicalAndOperation
from msit_opcheck.golden_funcs.pad import PadOperation
from msit_opcheck.golden_funcs.add_n import AddOperation
from msit_opcheck.golden_funcs.batchnorm import BatchNormOperation
from msit_opcheck.golden_funcs.cast import CastOperation
from msit_opcheck.golden_funcs.bias_add import BiasAddOperation
from msit_opcheck.golden_funcs.softmax import SoftmaxOperation 
from msit_opcheck.golden_funcs.mat_mul import MatmulOperation
from msit_opcheck.golden_funcs.bninference_d import BnInferenceOperation
from msit_opcheck.golden_funcs.concat_d import ConcatOperation
from msit_opcheck.golden_funcs.conv2d import Conv2dOperation
from msit_opcheck.golden_funcs.gather_v2 import GatherOperation
from msit_opcheck.golden_funcs.minimum import MinimumOperation
from msit_opcheck.golden_funcs.pack import PackOperation
from msit_opcheck.golden_funcs.reduce import ReduceSumOperation, ReduceMeanOperation
from msit_opcheck.golden_funcs.relu import ReluOperation
from msit_opcheck.golden_funcs.sigmoid import SigmoidOperation
from msit_opcheck.golden_funcs.tanh import TanhOperation
from msit_opcheck.golden_funcs.tile_d import TileDOperation
from msit_opcheck.golden_funcs.transpose import TransposeOperation
from msit_opcheck.golden_funcs.mul import MulOperation
from msit_opcheck.golden_funcs.select import SelectOperation
from msit_opcheck.golden_funcs.clip_by_value import ClipByValueOperation
from msit_opcheck.golden_funcs.rsqrt import RsqrtOperation
from msit_opcheck.golden_funcs.less import LessOperation
from msit_opcheck.golden_funcs.sub import SubOperation
from msit_opcheck.golden_funcs.stridedslice import StridedSliceOperation
from msit_opcheck.golden_funcs.batch_matmul import BatchMatMulOperation

OP_DICT = dict({"Pad": PadOperation, 
                "PadD": PadOperation, 
                "LogicalOr": LogicalOrOperation, 
                "Adds": AddOperation,
                "Add": AddOperation, 
                "BatchNorm": BatchNormOperation, 
                "Cast": CastOperation, 
                "SoftmaxV2": SoftmaxOperation, 
                "BiasAdd": BiasAddOperation, 
                "MatMulV2": MatmulOperation,
                "BNInfer": BnInferenceOperation,
                "ConcatV2": ConcatOperation,
                "Conv2D": Conv2dOperation,
                "GatherV2": GatherOperation,
                "LogicalAnd": LogicalAndOperation,
                "Minimum": MinimumOperation,
                "Pack": PackOperation,
                "ReduceSum": ReduceSumOperation,
                "ReduceMean": ReduceMeanOperation,
                "Relu": ReluOperation,
                "Sigmoid": SigmoidOperation,
                "Tanh": TanhOperation,
                "Tile": TileDOperation,
                "Transpose": TransposeOperation,
                "Mul": MulOperation,
                "Select": SelectOperation,
                "ClipByValue": ClipByValueOperation,
                "Rsqrt": RsqrtOperation,
                "Less": LessOperation,
                "Sub": SubOperation,
                "StridedSlice": StridedSliceOperation,
                "BatchMatMulV2": BatchMatMulOperation,
                })
