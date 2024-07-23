# Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
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

from enum import Enum
import torch
import torch_npu

from msit_llm.opcheck import operation_test
from msit_llm.common.log import logger


class CompressType(Enum):
    COMPRESS_TYPE_UNDEFINED = 0
    COMPRESS_TYPE_KVHEAD = 1


class OpcheckReshapeAndCacheOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        num_tokens, num_heads, head_size = in_tensors[0].shape
        data_type = in_tensors[0].dtype
        slot_mapping = in_tensors[4]
        soc_version = self.get_soc_version()
        if soc_version == "Ascend910B":
            num_blocks, block_size, _, _ = in_tensors[2].shape
            key_expect = in_tensors[2]
            value_expect = in_tensors[3]
            compress_type = self.op_param.get("compressType", CompressType.COMPRESS_TYPE_UNDEFINED)
            if compress_type == CompressType.COMPRESS_TYPE_KVHEAD:
                pass
        else:
            pass
            
        golden = []
        inplace_idx = self.case_info.get("inplace_idx", None)
        for index in inplace_idx:
            golden.append(in_tensors[index])
        return golden

    def test(self):
        ret = self.validate_param("inplace_idx")
        if not ret:
            return
        self.execute_inplace()