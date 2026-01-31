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
__all__ = ["KnowledgeBase", "KnowledgeFactory", "Pattern"]


from auto_optimizer.pattern.knowledges.knowledge_base import KnowledgeBase
from auto_optimizer.pattern.knowledge_factory import KnowledgeFactory

from auto_optimizer.pattern.knowledges import knowledge_conv1d2conv2d
from auto_optimizer.pattern.knowledges import knowledge_merge_consecutive_slice
from auto_optimizer.pattern.knowledges import knowledge_transpose_large_input_conv
from auto_optimizer.pattern.knowledges import knowledge_merge_consecutive_concat
from auto_optimizer.pattern.knowledges import knowledge_type_cast
from auto_optimizer.pattern.knowledges import knowledge_split_qkv_matmul
from auto_optimizer.pattern.knowledges import knowledge_split_large_kernel
from auto_optimizer.pattern.knowledges import knowledge_resize_mode_to_nearest
from auto_optimizer.pattern.knowledges import knowledge_topk_fix
from auto_optimizer.pattern.knowledges import knowledge_merge_casts
from auto_optimizer.pattern.knowledges import knowledge_empty_slice_fix
from auto_optimizer.pattern.knowledges import knowledge_dynamic_reshape
from auto_optimizer.pattern.knowledges import knowledge_gather_to_split
from auto_optimizer.pattern.knowledges import knowledge_avgpool_split
from auto_optimizer.pattern.knowledges import knowledge_bn_folding
from auto_optimizer.pattern.knowledges import knowledge_modify_reflection_pad

from auto_optimizer.pattern.pattern import Pattern