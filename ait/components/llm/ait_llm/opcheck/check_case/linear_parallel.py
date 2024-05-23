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
import torch.distributed as dist

from ait_llm.opcheck import operation_test
from ait_llm.common.log import logger


class OpcheckLinearParallelOperation(operation_test.OperationTest):
    def add_residual(self, data, in_tensors):
        has_residual = self.op_param.get("has_residual", False)
        if has_residual:
            data += in_tensors[2]
        return data

    def get_matmul_result(self, in_tensors):
        x = in_tensors[0]
        weight = in_tensors[1]
        trans_weight = self.op_param.get("transWeight", True)
        if trans_weight:
            if len(weight.shape) == 2:
                weight = torch.permute(weight, (1, 0))
            if len(weight.shape) == 3:
                weight = torch.permute(weight, (0, 2, 1))
        matmul_result = torch.matmul(x, weight)
        return matmul_result

    def get_quant_result(self, in_tensors, quant_type=-1, group_size=0, out_data_type=-1):
        is_quant_after = in_tensors[0].dtype == torch.int8 and in_tensors[1].dtype == torch.int8
        quant_tensor = self.get_matmul_result(in_tensors) if is_quant_after else in_tensors[1].to(torch.float32)

        try:
            bias = in_tensors[2]
        except:
            bias = None
        if bias is not None and bias.nelement() != 0:
            quant_tensor = quant_tensor.to(torch.float) + bias.to(torch.float)

        try:
            scale = in_tensors[3]
        except:
            scale = None
        if scale is not None:
            if quant_type == 2:
                dequantized_groups = [group * scale[i] for i, group in enumerate(quant_tensor.split(group_size, dim=0))]
                quant_tensor = torch.cat(dequantized_groups, dim=0)
            else:
                quant_tensor *= scale

        if is_quant_after: 
             result_dtype = torch.float16 if out_data_type == 1 else torch.bfloat16
             result = quant_tensor.to(result_dtype)
        else:
            in_tensors[1] = quant_tensor.to(in_tensors[0].dtype)
            result = self.get_matmul_result(in_tensors)

        return result

    def pure_linear(self, in_tensors, quant_type=-1, group_size=0, out_data_type=-1):
        if quant_type >= 0:
            result = self.get_quant_result(in_tensors, quant_type, group_size, out_data_type)
        else:
            result = self.get_matmul_result(in_tensors)
        return [result]

    def all_reduce(self, in_tensors, rank_size, quant_type=-1, group_size=0, out_data_type=-1):
        rank, rank_root, rank_size = self.get_rank_info()
        golden_result = torch.zeros_like(self.pure_linear(in_tensors, quant_type, group_size, out_data_type)[0])
        for i in range(rank_root, rank_size):
            new_in_tensors = self.get_in_tensors_from_single_device(i, rank) 
            linear_result = self.pure_linear(new_in_tensors, quant_type, group_size, out_data_type)[0]
            golden_result += linear_result
        golden_result = self.add_residual(golden_result, in_tensors)
        return [golden_result]

    def reduce_scatter(self, in_tensors, rank, rank_size):
        rank, rank_root, rank_size = self.get_rank_info()
        sum_tensor = torch.zeros_like(self.get_matmul_result(in_tensors))
        for i in range(rank_root, rank_size):
            new_in_tensors = self.get_in_tensors_from_single_device(i, rank) 
            matmul_result = self.get_matmul_result(new_in_tensors)
            sum_tensor += matmul_result
        chunks = torch.split(sum_tensor, int(in_tensors[0].shape[0] / rank_size))
        golden_result = chunks[rank]
        golden_result = self.add_residual(golden_result, in_tensors)
        return [golden_result]

    def all_gather_linear(self, in_tensors, rank_size):
        rank, rank_root, rank_size = self.get_rank_info()
        golden_mid_tensor = None
        for i in range(rank_root, rank_size):
            new_in_tensors = self.get_in_tensors_from_single_device(i, rank) 
            if golden_mid_tensor is None:
                golden_mid_tensor = new_in_tensors[0].clone()
            else:
                golden_mid_tensor = torch.cat((golden_mid_tensor, new_in_tensors[0]), dim=0)
        golden_result = self.get_matmul_result([golden_mid_tensor, in_tensors[1]])
        golden_result = self.add_residual(golden_result, in_tensors)

        keep_intermediate = self.op_param.get("keepIntermediate", False)
        if keep_intermediate:
            res = [golden_result, golden_mid_tensor]
        else:
            res = [golden_result]

        return res

    def golden_calc(self, in_tensors):
        backend = self.op_param.get('backend', None)
        rank = self.op_param.get('rank', None)
        rank_size = self.op_param.get("rankSize", None)

        if backend != "lcoc":
            golden_result = self.all_reduce(in_tensors, rank_size)
        else:
            cal_type = self.op_param.get("type", 0)
            quant_type = self.op_param.get("quantType", -1)
            group_size = self.op_param.get("quantGroupSize", 0)
            out_data_type = self.op_param.get("outDataType", -1)

            if cal_type == 0:
                golden_result = self.all_reduce(in_tensors, rank_size, quant_type, group_size, out_data_type)
            elif cal_type == 1:
                golden_result = self.reduce_scatter(in_tensors, rank, rank_size)
            elif cal_type == 2:
                golden_result = self.all_gather_linear(in_tensors, rank_size)
            elif cal_type == 3:
                golden_result = self.pure_linear(in_tensors, quant_type, group_size, out_data_type)

        return golden_result

    def test(self):
        soc_version = self.get_soc_version()
        if soc_version != 'Ascend910B':
            logger_text = "This case is only supported on Ascend910B!"
            logger.info(logger_text)
            return

        ret = self.validate_param("backend", "rank", "rankRoot", "rankSize")
        if not ret:
            return

        self.execute()