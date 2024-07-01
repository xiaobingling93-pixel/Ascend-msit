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

from msit_llm.opcheck.check_case.activation import OpcheckActivationOperation
from msit_llm.opcheck.check_case.all_gather import OpcheckAllGatherOperation
from msit_llm.opcheck.check_case.all_reduce import OpcheckAllReduceOperation
from msit_llm.opcheck.check_case.broadcast import OpcheckBroadcastOperation
from msit_llm.opcheck.check_case.concat import OpcheckConcatOperation
from msit_llm.opcheck.check_case.cumsum import OpcheckCumsumOperation
from msit_llm.opcheck.check_case.elewise import OpcheckElewiseAddOperation
from msit_llm.opcheck.check_case.fastsoftmax import OpcheckFastSoftMaxOperation
from msit_llm.opcheck.check_case.fastsoftmaxgrad import OpcheckFastSoftMaxGradOperation
from msit_llm.opcheck.check_case.fill import OpcheckFillOperation
from msit_llm.opcheck.check_case.gather import OpcheckGatherOperation
from msit_llm.opcheck.check_case.genattentionmask import OpcheckElewiseSubOperation
from msit_llm.opcheck.check_case.kv_cache import OpcheckKvCacheOperation
from msit_llm.opcheck.check_case.linear import OpcheckLinearOperation
from msit_llm.opcheck.check_case.linear_sparse import OpcheckLinearSparseOperation
from msit_llm.opcheck.check_case.matmul import OpcheckMatmulOperation
from msit_llm.opcheck.check_case.pad import OpcheckPadOperation
from msit_llm.opcheck.check_case.paged_attention import OpcheckPagedAttentionAttentionOperation
from msit_llm.opcheck.check_case.repeat import OpcheckRepeatOperation
from msit_llm.opcheck.check_case.reshape_and_cache import OpcheckReshapeAndCacheOperation
from msit_llm.opcheck.check_case.rms_norm import OpcheckRmsNormOperation
from msit_llm.opcheck.check_case.rope_grad import OpcheckRopeGradOperation
from msit_llm.opcheck.check_case.rope import OpcheckUnpadRopeOperation
from msit_llm.opcheck.check_case.self_attention import OpcheckUnpadSelfAttentionOperation
from msit_llm.opcheck.check_case.set_value import OpcheckSetValueOperation
from msit_llm.opcheck.check_case.slice import OpcheckSliceOperation
from msit_llm.opcheck.check_case.softmax import OpcheckSoftmaxOperation
from msit_llm.opcheck.check_case.sort import OpcheckSortOperation
from msit_llm.opcheck.check_case.split import OpcheckAddOperation
from msit_llm.opcheck.check_case.stridebatchmatmul import OpcheckStridedBatchMatmulOperation
from msit_llm.opcheck.check_case.topk_topp_sampling import OpcheckToppOperation
from msit_llm.opcheck.check_case.transpose import OpcheckTransposeOperation
from msit_llm.opcheck.check_case.unpad import OpcheckUnpadOperation
from msit_llm.opcheck.check_case.as_strided import OpcheckAsStridedOperation
from msit_llm.opcheck.check_case.layer_norm import OpcheckLayerNormOperation
from msit_llm.opcheck.check_case.linear_parallel import OpcheckLinearParallelOperation
from msit_llm.opcheck.check_case.multinomial import OpcheckMultinomialOperation
from msit_llm.opcheck.check_case.reduce import OpcheckReduceOperation
from msit_llm.opcheck.check_case.transdata import OpcheckTransdataOperation
from msit_llm.opcheck.check_case.where import OpcheckWhereOperation


OP_NAME_DICT = dict({
    "ActivationOperation":OpcheckActivationOperation,
    "AllGatherOperation":OpcheckAllGatherOperation,
    "AllReduceOperation":OpcheckAllReduceOperation,
    "BroadcastOperation":OpcheckBroadcastOperation,
    "ConcatOperation":OpcheckConcatOperation,
    "CumsumOperation":OpcheckCumsumOperation,
    "ElewiseOperation":OpcheckElewiseAddOperation,
    "FastSoftMaxOperation":OpcheckFastSoftMaxOperation,
    "FastSoftMaxGradOperation":OpcheckFastSoftMaxGradOperation,
    "FillOperation":OpcheckFillOperation,
    "GatherOperation":OpcheckGatherOperation,
    "GenAttentionMaskOperation":OpcheckElewiseSubOperation,
    "KvCacheOperation":OpcheckKvCacheOperation,
    "LinearOperation":OpcheckLinearOperation,
    "LinearSparseOperation":OpcheckLinearSparseOperation,
    "MatmulOperation":OpcheckMatmulOperation,
    "PadOperation":OpcheckPadOperation,
    "PagedAttentionOperation":OpcheckPagedAttentionAttentionOperation,
    "RepeatOperation":OpcheckRepeatOperation,
    "ReshapeAndCacheOperation":OpcheckReshapeAndCacheOperation,
    "RmsNormOperation":OpcheckRmsNormOperation,
    "RopeOperation":OpcheckUnpadRopeOperation,
    "RopeGradOperation":OpcheckRopeGradOperation,
    "SelfAttentionOperation":OpcheckUnpadSelfAttentionOperation,
    "SetValueOperation":OpcheckSetValueOperation,
    "SliceOperation":OpcheckSliceOperation,
    "SoftmaxOperation":OpcheckSoftmaxOperation,
    "SortOperation":OpcheckSortOperation,
    "SplitOperation":OpcheckAddOperation,
    "StridedBatchMatmulOperation":OpcheckStridedBatchMatmulOperation,
    "TopkToppSamplingOperation":OpcheckToppOperation,
    "TransposeOperation":OpcheckTransposeOperation,
    "UnpadOperation":OpcheckUnpadOperation,
    "AsStridedOperation":OpcheckAsStridedOperation,
    "LayerNormOperation":OpcheckLayerNormOperation,
    "LinearParallelOperation":OpcheckLinearParallelOperation,
    "MultinomialOperation":OpcheckMultinomialOperation,
    "ReduceOperation":OpcheckReduceOperation,
    "TransdataOperation":OpcheckTransdataOperation,
    "WhereOperation":OpcheckWhereOperation,
})