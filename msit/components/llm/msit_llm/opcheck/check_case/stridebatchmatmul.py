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
import torch

from msit_llm.opcheck import operation_test


class OpcheckStridedBatchMatmulOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        a = in_tensors[0].flatten()
        b = in_tensors[1].flatten()

        batch_start_a = 0
        batch_start_b = 0
        batch_start_c = 0
        list_a = []
        list_b = []

        batch = self.op_param.get("batch", None)
        head_num = self.op_param.get("head_num", None)
        trans_a = self.op_param.get("trans_a", None)
        trans_b = self.op_param.get("trans_b", None)
        m, n, k = self.op_param.get("m", None), self.op_param.get("n", None), self.op_param.get("k", None)
        lda, ldb, ldc = self.op_param.get("lda", None), self.op_param.get("ldb", None), self.op_param.get("ldc", None)
        stridea = self.op_param.get("strideA", None)
        strideb = self.op_param.get("strideB", None)
        stridec = self.op_param.get("strideC", None)

        c = torch.zeros(sum([m[i] * n[i] for i in range(batch)]) * head_num, dtype=torch.float16, device=a.device)

        for i in range(batch):
            for j in range(head_num):
                list_a = []
                list_b = []
                row_a = m[i] if not trans_a else k[i]
                col_a = k[i] if not trans_a else m[i]
                for t in range(row_a):
                    start_a = lda[i] * t + stridea[i] * j + batch_start_a
                    end_a = start_a + col_a
                    list_a.append(a[start_a:end_a])
                row_b = k[i] if not trans_b else n[i]
                col_b = n[i] if not trans_b else k[i]
                for t in range(row_b):
                    start_b = ldb[i] * t + strideb[i] * j + batch_start_b
                    end_b = start_b + col_b
                    list_b.append(b[start_b:end_b])
                mat_a = torch.stack(list_a)
                mat_b = torch.stack(list_b)
                mat_a = torch.transpose(mat_a, 0, 1) if trans_a else mat_a
                mat_b = torch.transpose(mat_b, 0, 1) if trans_b else mat_b
                mat_c = torch.matmul(mat_a, mat_b).half()
                for t in range(mat_c.shape[0]):
                    start_c = ldc[i] * t + stridec[i] * j + batch_start_c
                    end_c = start_c + mat_c.shape[1]
                    c[start_c:end_c] = mat_c[t, :]
            batch_start_a += m[i] * k[i] * head_num
            batch_start_b += n[i] * k[i] * head_num
            batch_start_c += m[i] * n[i] * head_num
        return [c]

    def test_add_bmm1(self):
        self.execute()