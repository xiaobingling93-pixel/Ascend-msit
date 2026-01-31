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
__all__ = ["PreProcessBase", "PostProcessBase", "EvaluateBase", "InferenceBase"]


from auto_optimizer.inference_engine.pre_process.pre_process_base import PreProcessBase
from auto_optimizer.inference_engine.post_process.post_process_base import PostProcessBase
from auto_optimizer.inference_engine.evaluate.evaluate_base import EvaluateBase
from auto_optimizer.inference_engine.inference.inference_base import InferenceBase