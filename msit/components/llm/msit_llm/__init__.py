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
from msit_llm.common.tool import read_atb_data, seed_all
from msit_llm.compare.cmp_utils import compare_data
from msit_llm.common.json_fitter import atb_json_to_onnx
from msit_llm.dump.torch_dump import DumpConfig
from msit_llm.dump.torch_dump import register_hook
from msit_llm.metrics.case_filter import CaseFilter
from msit_llm.bc_analyze.analyzer import Analyzer
from msit_llm.bc_analyze.synthezier import Synthesizer
