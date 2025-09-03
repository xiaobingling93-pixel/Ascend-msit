# -*- coding: utf-8 -*-
# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
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

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

from msserviceprofiler.modelevalstate.config.config import CommunicationConfig, map_param_with_value
from msserviceprofiler.modelevalstate.config.config import settings, default_support_field
from msserviceprofiler.modelevalstate.optimizer.communication import CommunicationForFile, CustomCommand
from msserviceprofiler.modelevalstate.optimizer.optimizer import ScheduleWithMultiMachine


class TestScheduleWithMultiMachine:

    def test_schedule_with_multi_machine_init(self):
        # 创建模拟对象
        mock_communication_config = MagicMock(spec=CommunicationConfig)
        mock_communication = MagicMock(spec=CommunicationForFile)
        mock_custom_command = MagicMock(spec=CustomCommand)

        # 模拟CommunicationForFile和CustomCommand的构造函数
        with patch('msserviceprofiler.modelevalstate.optimizer.scheduler.CommunicationForFile',
                   return_value=mock_communication) as mock_communication_class, \
                patch('msserviceprofiler.modelevalstate.optimizer.scheduler.CustomCommand',
                      return_value=mock_custom_command) as mock_custom_command_class:
            mock_communication_config.cmd_file = None
            mock_communication_config.res_file = None
            # 创建ScheduleWithMultiMachine实例
            scheduler = ScheduleWithMultiMachine(mock_communication_config, MagicMock(), MagicMock(), MagicMock())

            # 验证CommunicationForFile和CustomCommand是否被正确初始化
            mock_communication_class.assert_called_once_with(mock_communication_config.cmd_file,
                                                             mock_communication_config.res_file)
            mock_custom_command_class.assert_called_once()

            # 验证communication属性是否被正确设置
            assert scheduler.communication_config == mock_communication_config
            assert scheduler.communication == mock_communication

            # 验证cmd属性是否被正确设置
            assert scheduler.cmd == mock_custom_command

    @pytest.fixture
    def schedule_with_multi_machine(self):
        mock_communication_config = MagicMock(spec=CommunicationConfig)
        mock_communication = MagicMock(spec=CommunicationForFile)
        mock_custom_command = MagicMock(spec=CustomCommand)
        mock_communication_config.cmd_file = None
        mock_communication_config.res_file = None
        with patch('msserviceprofiler.modelevalstate.optimizer.scheduler.CommunicationForFile',
                   return_value=mock_communication, autospec=True) as mock_communication_class, \
                patch('msserviceprofiler.modelevalstate.optimizer.scheduler.CustomCommand',
                      return_value=mock_custom_command, autospec=True) as mock_custom_command_class:
            schedule = ScheduleWithMultiMachine(settings.communication, MagicMock(), MagicMock(), MagicMock())
            schedule.cmd = MagicMock()
            schedule.communication = MagicMock()
            yield schedule

    @patch('msserviceprofiler.modelevalstate.optimizer.scheduler.get_train_sub_path')
    def test_set_back_up_path_with_bak_path(self, mock_get_train_sub_path, schedule_with_multi_machine, tmpdir):
        # Arrange
        bak_path = Path(tmpdir)
        schedule_with_multi_machine.bak_path = bak_path
        mock_get_train_sub_path.return_value = tmpdir
        # Act
        schedule_with_multi_machine.set_back_up_path()

        # Assert
        mock_get_train_sub_path.assert_called_once_with(bak_path)
        schedule_with_multi_machine.simulator.bak_path == bak_path
        schedule_with_multi_machine.benchmark.bak_path == bak_path
        schedule_with_multi_machine.communication.send_command.assert_called_once_with(
            f'{schedule_with_multi_machine.cmd.backup} params:{bak_path}')
        schedule_with_multi_machine.communication.clear_command.assert_called_once_with(
            f'{schedule_with_multi_machine.cmd.backup} params:{bak_path}')

    @patch('msserviceprofiler.modelevalstate.optimizer.scheduler.time.sleep', return_value=None)
    def test_monitoring_status_success(self, mock_sleep, schedule_with_multi_machine):
        type(schedule_with_multi_machine.cmd).process_poll = PropertyMock(return_value="mocked process poll")
        schedule_with_multi_machine.communication.send_command = MagicMock()
        schedule_with_multi_machine.simulator.process.poll = MagicMock(return_value=None)
        schedule_with_multi_machine.communication.clear_command = MagicMock(return_value=None)
        schedule_with_multi_machine.benchmark.check_success = MagicMock(return_value=True)
        schedule_with_multi_machine.stop_target_server = MagicMock()

        schedule_with_multi_machine.monitoring_status()

        schedule_with_multi_machine.communication.send_command.assert_called_with("mocked process poll")
        schedule_with_multi_machine.simulator.process.poll.assert_called()
        schedule_with_multi_machine.communication.clear_command.assert_called_with("mocked process poll")
        schedule_with_multi_machine.benchmark.check_success.assert_called()
        schedule_with_multi_machine.stop_target_server.assert_not_called()

    @patch('msserviceprofiler.modelevalstate.optimizer.scheduler.time.sleep', return_value=None)
    def test_monitoring_status_failure(self, mock_sleep, schedule_with_multi_machine):
        type(schedule_with_multi_machine.cmd).process_poll = PropertyMock(return_value='mocked process poll')
        schedule_with_multi_machine.communication.send_command = MagicMock()
        schedule_with_multi_machine.simulator.process.poll = MagicMock(return_value=1)
        schedule_with_multi_machine.communication.clear_command = MagicMock(return_value=None)
        schedule_with_multi_machine.benchmark.check_success = MagicMock(return_value=False)
        schedule_with_multi_machine.stop_target_server = MagicMock()

        with pytest.raises(subprocess.SubprocessError):
            schedule_with_multi_machine.monitoring_status()

        schedule_with_multi_machine.communication.send_command.assert_called_with('mocked process poll')
        schedule_with_multi_machine.simulator.process.poll.assert_called()
        schedule_with_multi_machine.communication.clear_command.assert_called_with('mocked process poll')
        schedule_with_multi_machine.benchmark.check_success.assert_not_called()
        schedule_with_multi_machine.stop_target_server.assert_called()

    def test_run_simulate(self, schedule_with_multi_machine):
        # 模拟方法
        schedule_with_multi_machine.cmd = CustomCommand()
        schedule_with_multi_machine.benchmark.prepare.return_value = None
        schedule_with_multi_machine.communication.send_command.return_value = None
        schedule_with_multi_machine.communication.clear_command.return_value = None
        schedule_with_multi_machine.simulator.run.return_value = None
        schedule_with_multi_machine.wait_simulate = MagicMock(return_value=None)
        _params = np.random.random(len(default_support_field))
        # 调用run_simulate方法
        schedule_with_multi_machine.simulate_run_info = map_param_with_value(_params, default_support_field)
        schedule_with_multi_machine.run_simulate(_params, default_support_field)

        # 验证方法调用
        schedule_with_multi_machine.benchmark.prepare.assert_called_once()
        assert schedule_with_multi_machine.communication.send_command.call_count == 2
        assert schedule_with_multi_machine.communication.clear_command.call_count == 2
        schedule_with_multi_machine.wait_simulate.assert_called_once()

    @patch('msserviceprofiler.modelevalstate.optimizer.scheduler.Scheduler.stop_target_server')
    def test_stop_target_server_with_del_log(self, mock_super_stop, schedule_with_multi_machine):
        # 测试当del_log为True时的行为
        schedule_with_multi_machine.cmd = CustomCommand()
        schedule_with_multi_machine.stop_target_server(del_log=True)

        # 验证Scheduler.stop_target_server被调用
        mock_super_stop.assert_called_once_with(True)

        # 验证communication.send_command和communication.clear_command被正确调用
        assert "params:True" in schedule_with_multi_machine.cmd.history[-1]
        schedule_with_multi_machine.communication.send_command.assert_called_once_with(
            f"{schedule_with_multi_machine.cmd.history[-1]}")
        schedule_with_multi_machine.communication.clear_command.assert_called_once_with(
            f"{schedule_with_multi_machine.cmd.history[-1]}")
