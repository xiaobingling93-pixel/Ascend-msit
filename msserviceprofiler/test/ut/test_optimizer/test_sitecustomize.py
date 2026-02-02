# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2025-2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          `http://license.coscl.org.cn/MulanPSL2`
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

import os
import unittest
from unittest.mock import patch, MagicMock
from msserviceprofiler.modelevalstate.sitecustomize import dispatch, MODEL_EVAL_STATE_SIMULATE, MODEL_EVAL_STATE_ALL


class TestDispatch(unittest.TestCase):
    
    @patch('msserviceprofiler.modelevalstate.sitecustomize.logger')
    @patch('msserviceprofiler.modelevalstate.patch.enable_patch')
    def test_dispatch_simulate_true(self, mock_enable_patch, mock_logger):
        # Set environment variable for simulation
        os.environ[MODEL_EVAL_STATE_SIMULATE] = 'True'
        dispatch(MODEL_EVAL_STATE_SIMULATE)
        mock_logger.info.assert_called_with(f"The collected patch is successfully installed.")
        mock_enable_patch.assert_called_with(MODEL_EVAL_STATE_SIMULATE)

    @patch('msserviceprofiler.modelevalstate.sitecustomize.logger')
    @patch('msserviceprofiler.modelevalstate.patch.enable_patch')
    def test_dispatch_all_true(self, mock_enable_patch, mock_logger):
        # Set environment variable for optimization
        os.environ[MODEL_EVAL_STATE_ALL] = 'true'
        dispatch(MODEL_EVAL_STATE_ALL)
        mock_logger.info.assert_called_with(f"The collected patch is successfully installed.")
        mock_enable_patch.assert_called_with(MODEL_EVAL_STATE_ALL)

    @patch('msserviceprofiler.modelevalstate.sitecustomize.logger')
    def test_dispatch_simulate_false(self, mock_logger):
        # Set environment variable for simulation to false
        os.environ[MODEL_EVAL_STATE_SIMULATE] = 'False'
        dispatch(MODEL_EVAL_STATE_SIMULATE)
        mock_logger.debug.assert_called_with(f"{MODEL_EVAL_STATE_SIMULATE}: False")

    @patch('msserviceprofiler.modelevalstate.sitecustomize.logger')
    def test_dispatch_all_false(self, mock_logger):
        # Set environment variable for optimization to false
        os.environ[MODEL_EVAL_STATE_ALL] = 'false'
        dispatch(MODEL_EVAL_STATE_ALL)
        mock_logger.debug.assert_called_with(f"{MODEL_EVAL_STATE_ALL}: false")

    @patch('msserviceprofiler.modelevalstate.sitecustomize.logger')
    def test_dispatch_simulate_not_set(self, mock_logger):
        # Do not set environment variable for simulation
        if MODEL_EVAL_STATE_SIMULATE in os.environ:
            del os.environ[MODEL_EVAL_STATE_SIMULATE]
        dispatch(MODEL_EVAL_STATE_SIMULATE)
        mock_logger.debug.assert_called_with(f"{MODEL_EVAL_STATE_SIMULATE}: None")

    @patch('msserviceprofiler.modelevalstate.sitecustomize.logger')
    def test_dispatch_all_not_set(self, mock_logger):
        # Do not set environment variable for optimization
        if MODEL_EVAL_STATE_ALL in os.environ:
            del os.environ[MODEL_EVAL_STATE_ALL]
        dispatch(MODEL_EVAL_STATE_ALL)
        mock_logger.debug.assert_called_with(f"{MODEL_EVAL_STATE_ALL}: None")