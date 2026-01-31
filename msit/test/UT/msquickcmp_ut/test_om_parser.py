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
import os
import pytest
from msquickcmp.npu.om_parser import OmParser


@pytest.fixture(scope="module", autouse=True)
def om_parser() -> OmParser:
    ut_dir = os.path.dirname(os.path.realpath(__file__))
    om_parser = OmParser(os.path.join(ut_dir, "..", "resource", "msquickcmp", "om", "model.json"))
    return om_parser


def test_dynamic_scenario(om_parser):
    is_dynamic_scenario, _ = om_parser.get_dynamic_scenario_info()
    assert is_dynamic_scenario is False


def test_get_shape_size(om_parser):
    shape_size_array = om_parser.get_shape_size()
    assert shape_size_array == [1280000, 320000]


def test_net_output_count(om_parser):
    count = om_parser.get_net_output_count()
    assert count == 3

    atc_cmd = om_parser.get_atc_cmdline()
    assert "model" in atc_cmd

    net_output = om_parser.get_expect_net_output_name()
    assert len(net_output) == 3
    assert net_output.get(0) == "Cast_1219:0:output0"
    assert net_output.get(1) == 'PartitionedCall_Gather_1221_gatherv2_3:0:output2'
    assert net_output.get(2) == 'Reshape_1213:0:output1'


def test_get_atc_cmdline(om_parser):
    atc_cmd = om_parser.get_atc_cmdline()
    assert "model" in atc_cmd


def test_get_expect_net_output_name(om_parser):
    net_output = om_parser.get_expect_net_output_name()
    assert len(net_output) == 3
    assert net_output.get(0) == "Cast_1219:0:output0"
    assert net_output.get(1) == 'PartitionedCall_Gather_1221_gatherv2_3:0:output2'
    assert net_output.get(2) == 'Reshape_1213:0:output1'
