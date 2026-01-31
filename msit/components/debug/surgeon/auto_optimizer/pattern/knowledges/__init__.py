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
from auto_optimizer.pattern.knowledges.knowledge_conv1d2conv2d import KnowledgeConv1d2Conv2d
from auto_optimizer.pattern.knowledges.knowledge_merge_consecutive_slice import KnowledgeMergeConsecutiveSlice
from auto_optimizer.pattern.knowledges.knowledge_transpose_large_input_conv import KnowledgeTransposeLargeInputConv
from auto_optimizer.pattern.knowledges.knowledge_merge_consecutive_concat import KnowledgeMergeConsecutiveConcat
from auto_optimizer.pattern.knowledges.knowledge_type_cast import KnowledgeTypeCast
from auto_optimizer.pattern.knowledges.knowledge_split_qkv_matmul import KnowledgeSplitQKVMatmul
from auto_optimizer.pattern.knowledges.knowledge_split_large_kernel import KnowledgeSplitLargeKernelConv
from auto_optimizer.pattern.knowledges.knowledge_resize_mode_to_nearest import KnowledgeResizeModeToNearest
from auto_optimizer.pattern.knowledges.knowledge_topk_fix import KnowledgeTopkFix
from auto_optimizer.pattern.knowledges.knowledge_merge_casts import KnowledgeMergeCasts
from auto_optimizer.pattern.knowledges.knowledge_empty_slice_fix import KnowledgeEmptySliceFix
from auto_optimizer.pattern.knowledges.knowledge_dynamic_reshape import KnowledgeDynamicReshape
from auto_optimizer.pattern.knowledges.knowledge_gather_to_split import KnowledgeGatherToSplit
from auto_optimizer.pattern.knowledges.knowledge_avgpool_split import KnowledgeAvgPoolSplit
from auto_optimizer.pattern.knowledges.knowledge_bn_folding import KnowledgeBNFolding
from auto_optimizer.pattern.knowledges.knowledge_modify_reflection_pad import KnowledgeModifyReflectionPad
from auto_optimizer.pattern.knowledges.big_kernel.knowledge_big_kernel import KnowledgeBigKernel
