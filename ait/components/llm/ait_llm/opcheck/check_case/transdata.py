# Copyright (c) 2023-2023 Huawei Technologies Co., Ltd.
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

import torch
import torch_npu

from ait_llm.opcheck import operation_test
from ait_llm.common.log import logger


class OpcheckTransdataOperation(operation_test.OperationTest):
    @staticmethod
    def round_up(x, align):
        if align == 0:
            return -1
        return (x + align - 1) // align * align
    
    @staticmethod
    def custom_pad(x, pad_dims):
        return torch.nn.functional.pad(x, pad_dims)

    @staticmethod
    def custom_reshape(x, target_shape):
        return x.reshape(target_shape)
    
    @staticmethod
    def custom_transpose(x, dim1, dim2):
        return x.transpose(dim1, dim2)

    def golden_nd_to_nz_3d(self, in_tensors):
        align_int8 = 32
        default_align = 16

        aux_dims = [0, 0, 0, 0]
        aux_dims[0] = in_tensors[0].size(0)
        aux_dims[1] = self.round_up(in_tensors[0].size(1), default_align)

        pad_dims = [0, 0, 0, 0]  
        pad_dims[3] = self.round_up(in_tensors[0].size(1), default_align) - in_tensors[0].size(1)

        if in_tensors[0].dtype == torch.int8:
            aux_dims[2] = self.round_up(in_tensors[0].size(2), align_int8) // align_int8
            aux_dims[3] = align_int8
            pad_dims[1] = self.round_up(in_tensors[0].size(2), align_int8) - in_tensors[0].size(2)
        else:
            aux_dims[2] = self.round_up(in_tensors[0].size(2), default_align) // default_align
            aux_dims[3] = default_align
            pad_dims[1] = self.round_up(in_tensors[0].size(2), default_align) - in_tensors[0].size(2)
        
        return self.custom_transpose(
                    self.custom_reshape(
                        self.custom_pad(in_tensors[0], pad_dims),
                        aux_dims
                    ),
                    1, 2
                ).contiguous()

    def golden_nd_to_nz_2d(self, in_tensors):
        align_int8 = 32
        default_align = 16
        
        aux_dims = [0, 0, 0, 0]
        aux_dims[0] = 1
        aux_dims[1] = self.round_up(in_tensors[0].size(0), default_align)
 
        pad_dims = [0, 0, 0, 0]  
        pad_dims[3] = self.round_up(in_tensors[0].size(0), default_align) - in_tensors[0].size(0)
 
        if in_tensors[0].dtype == torch.int8:
            aux_dims[2] = self.round_up(in_tensors[0].size(1), align_int8) // align_int8
            aux_dims[3] = align_int8
            pad_dims[1] = self.round_up(in_tensors[0].size(1), align_int8) - in_tensors[0].size(1)
        else:
            aux_dims[2] = self.round_up(in_tensors[0].size(1), default_align) // default_align
            aux_dims[3] = default_align
            pad_dims[1] = self.round_up(in_tensors[0].size(1), default_align) - in_tensors[0].size(1)
        
        return self.custom_transpose(
                    self.custom_reshape(
                        self.custom_pad(in_tensors[0], pad_dims),
                        aux_dims
                    ),
                    1, 2
                ).contiguous()

    def golden_nz_to_nd(self, in_tensors, out_crops):
        aux_dims = [0, 0, 0]
        aux_dims[0] = in_tensors[0].size(0)
        aux_dims[1] = in_tensors[0].size(2)
        aux_dims[2] = in_tensors[0].size(1) * in_tensors[0].size(3)
        
        return self.custom_reshape(
                    self.custom_transpose(in_tensors[0], 1, 2),
                    aux_dims
                )[:, :out_crops[0], :out_crops[1]]

    def golden_calc(self, in_tensors):
        transdata_type = self.op_param.get("transdataType", None)
        if transdata_type == 2:
            if len(in_tensors[0].size()) == 3:
                golden_result = self.golden_nd_to_nz_3d(in_tensors)
            else:
                golden_result = self.golden_nd_to_nz_2d(in_tensors)
        else:
            out_crops = self.op_param.get("outCrops", None)
            golden_result = self.golden_nz_to_nd(in_tensors, out_crops)

        return [golden_result]

    def test(self):
        ret = self.validate_param("transdataType")
        if not ret:
            return
        self.execute()