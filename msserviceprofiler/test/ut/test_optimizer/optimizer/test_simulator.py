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
import os
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock
import shutil
import yaml
import pytest
import requests

import msserviceprofiler.modelevalstate
from msserviceprofiler.modelevalstate.config.config import get_settings, OptimizerConfigField, KubectlConfig, \
        MindieConfig
from msserviceprofiler.modelevalstate.optimizer.simulator import Simulator, VllmSimulator, enable_simulate_old,\
      DisaggregationSimulator
from msserviceprofiler.msguard import GlobalConfig


class TestSimulate(unittest.TestCase):

    def test_set_config_dict(self):
        origin_config = {"a": {"b": {"c": 3}}}
        Simulator.set_config(origin_config, "a.b.c", 4)
        assert origin_config["a"]["b"]["c"] == 4

    def test_set_config_list(self):
        origin_config = {"a": {"b": [{"c": 3}]}}
        Simulator.set_config(origin_config, "a.b.0.c", 4)
        assert origin_config["a"]["b"][0]["c"] == 4

    def test_set_config_new_key(self):
        origin_config = {"a": {"b": [{"c": 3}]}}
        Simulator.set_config(origin_config, "a.b.0.d", 4)
        assert origin_config["a"]["b"][0]["d"] == 4

    def test_set_config_add_dict_list_dict(self):
        origin_config = {"a": {"b": {"c": 3}}}
        Simulator.set_config(origin_config, "a.d.0.c", 4)
        assert origin_config["a"]["d"][0]["c"] == 4

    def test_set_config_add_dict(self):
        origin_config = {"a": {"b": [{"c": 3}]}}
        Simulator.set_config(origin_config, "a.b.1.c", 4)
        assert origin_config["a"]["b"][1]["c"] == 4

    def test_set_config_add_dict_list_dict_dict(self):
        origin_config = {"a": {"b": [{"c": 3}]}}
        Simulator.set_config(origin_config, "a.d.0.c.e", 4)
        assert origin_config["a"]["d"][0]["c"]["e"] == 4
    
    def test_is_int(self):
        # 测试is_int静态方法
        self.assertTrue(Simulator.is_int(1))
        self.assertTrue(Simulator.is_int("1"))
        self.assertFalse(Simulator.is_int("a"))


class TestVllmSimulator(unittest.TestCase):

    def setUp(self):
        # 创建测试对象
        self.vllm_config = MagicMock()
        self.vllm_config.process_name = "vllm_test"
        with patch('shutil.which', return_value="/path/to/vllm"):
            self.simulator = VllmSimulator(self.vllm_config)
        self.simulator.process = MagicMock()
        self.simulator.command = "mock_vllm_command"
        self.simulator.run_log = "mock_log_content"
    
    @patch('msserviceprofiler.modelevalstate.optimizer.simulator.logger')
    def test_check_success_with_startup_complete_in_log(self, mock_logger):
        """测试日志中包含'Application startup complete.'时返回True"""
        # 模拟get_log返回包含特定字符串的日志
        self.simulator.get_log = MagicMock(return_value="Application startup complete. Service is running.")
        
        # 调用被测方法
        result = self.simulator.check_success()
        
        # 验证结果
        self.assertTrue(result)
    
    @patch('msserviceprofiler.modelevalstate.optimizer.simulator.logger')
    def test_check_success_with_print_log_enabled(self, mock_logger):
        """测试开启print_log时记录日志"""
        # 设置print_log为True
        self.simulator.print_log = True
        self.simulator.get_log = MagicMock(return_value="Application startup complete.")
        
        # 调用被测方法
        result = self.simulator.check_success(print_log=True)
        
        # 验证结果和日志记录
        self.assertTrue(result)
        mock_logger.debug.assert_called_with("Application startup complete.")
    
    def test_check_success_process_running(self):
        """测试进程正在运行但日志中没有特定字符串时返回False"""
        # 模拟get_log返回不包含特定字符串的日志
        self.simulator.get_log = MagicMock(return_value="Server starting...")
        # 模拟进程正在运行(poll返回None)
        self.simulator.process.poll.return_value = None
        
        # 调用被测方法
        result = self.simulator.check_success()
        
        # 验证结果
        self.assertFalse(result)
    
    def test_check_success_process_exited_normally(self):
        """测试进程已完成且返回码为0时返回True"""
        # 模拟get_log返回不包含特定字符串的日志
        self.simulator.get_log = MagicMock(return_value="Server stopped normally.")
        # 模拟进程正常退出(poll返回0)
        self.simulator.process.poll.return_value = 0
        
        # 调用被测方法
        result = self.simulator.check_success()
        
        # 验证结果
        self.assertTrue(result)
    
    def test_check_success_process_exited_with_error(self):
        """测试进程已完成但返回码非0时抛出异常"""
        # 模拟get_log返回不包含特定字符串的日志
        self.simulator.get_log = MagicMock(return_value="Error occurred.")
        # 模拟进程异常退出(poll返回非0值)
        self.simulator.process.poll.return_value = 1
        self.simulator.process.returncode = 1
        
        # 验证抛出异常
        with self.assertRaises(subprocess.SubprocessError) as context:
            self.simulator.check_success()
        
        # 验证异常信息
        self.assertIn("Failed in run mock_vllm_command", str(context.exception))
        self.assertIn("return code: 1", str(context.exception))
    
    def test_check_success_empty_log(self):
        """测试空日志的情况"""
        # 模拟get_log返回空日志
        self.simulator.get_log = MagicMock(return_value="")
        # 模拟进程正在运行
        self.simulator.process.poll.return_value = None
        
        # 调用被测方法
        result = self.simulator.check_success()
        
        # 验证结果
        self.assertFalse(result)


def test_enable_simulate_with_simulator(tmpdir, monkeypatch):
    config_path = Path(tmpdir).joinpath("config.json")
    with open(config_path, 'w') as f:
        f.write("""{
    "Version": "1.0.0",
    "ServerConfig": {
        "tlsCaFile": [
            "ca.pem"
        ],
        "tlsCert": "security/certs/server.pem"
    },
    "BackendConfig": {
        "backendName": "mindieservice_llm_engine",
        "ModelDeployConfig": {
            "maxSeqLen": 2560,
            "maxInputTokenLen": 2048,
            "truncation": false,
            "ModelConfig": [
                {
                    "modelInstanceType": "Standard"
                }
            ]
        },
        "ScheduleConfig": {
            "templateType": "Standard"
        }
    }
}""")
    get_settings().mindie.config_path = config_path
    get_settings().mindie.config_bak_path = Path(tmpdir).joinpath("config_bak.json")
    simulator = Simulator(get_settings().mindie)
    monkeypatch.setattr(msserviceprofiler.modelevalstate.optimizer.simulator, "simulate_flag", True)
    with enable_simulate_old(simulator) as flag:
        with open(config_path, 'r') as f:
            data = json.load(f)
            assert data["BackendConfig"]["ModelDeployConfig"]["ModelConfig"][0][
                       "plugin_params"] == '{"plugin_type": "simulate"}'
    with open(config_path, 'r') as f:
        data = json.load(f)
        assert "plugin_params" not in data["BackendConfig"]["ModelDeployConfig"]["ModelConfig"][0]


def test_enable_simulate_with_simulator_plugin_params_exists(tmpdir, monkeypatch):
    config_path = Path(tmpdir).joinpath("config.json")
    data = {
        "BackendConfig": {
            "backendName": "mindieservice_llm_engine",
            "ModelDeployConfig": {
                "maxSeqLen": 2560,
                "ModelConfig": [
                    {
                        "modelInstanceType": "Standard",
                        "plugin_params": "{\"plugin_type\":\"tp\"}"
                    }
                ]

            },
            "ScheduleConfig": {
                "templateType": "Standard"
            }
        }
    }
    with open(config_path, 'w') as f:
        json.dump(data, f)
    get_settings().mindie.config_path = config_path
    get_settings().mindie.config_bak_path = Path(tmpdir).joinpath("config_bak.json")
    simulator = Simulator(get_settings().mindie)
    monkeypatch.setattr(msserviceprofiler.modelevalstate.optimizer.simulator, "simulate_flag", True)
    with enable_simulate_old(simulator) as flag:
        with open(config_path, 'r') as f:
            data = json.load(f)
            assert data["BackendConfig"]["ModelDeployConfig"]["ModelConfig"][0][
                       "plugin_params"] == '{"plugin_type": "tp,simulate"}'
    with open(config_path, 'r') as f:
        data = json.load(f)
        assert data["BackendConfig"]["ModelDeployConfig"]["ModelConfig"][0]["plugin_params"] == '{"plugin_type":"tp"}'


class TestDisaggregationSimulator(unittest.TestCase):
    def setUp(self):
        # 创建临时测试环境
        self.test_dir = Path("conf")
        self.yaml_dir = self.test_dir / "deployment"
        self.test_dir.mkdir(exist_ok=True)
        self.yaml_dir.mkdir(exist_ok=True)
        self.config_single_path = self.test_dir / "config.json"
        self.temp_file = tempfile.NamedTemporaryFile(delete=False)
        self.temp_file.write(b"MindIE-MS coordinator is ready!!!")
        self.temp_file.close()
        data = {
            "BackendConfig": {
                "backendName": "mindieservice_llm_engine",
                "ModelDeployConfig": {
                    "maxSeqLen": 2560,
                    "ModelConfig": [
                        {
                            "modelInstanceType": "Standard",
                            "plugin_params": "{\"plugin_type\":\"tp\"}"
                        }
                    ]

                },
                "ScheduleConfig": {
                    "templateType": "Standard"
                }
            }
        }
        with open(self.config_single_path, 'w') as f:
            json.dump(data, f)
        pd_data = {
            "default_p_rate": 1,
            "default_d_rate": 3
        }
        self.kubectl_single_path = self.test_dir / "deploy.sh"
        self.config_single_pd_path = self.test_dir / "ms_controller.json"
        self.yaml_path = self.yaml_dir / "mindie_service_single_container.yaml"
        service_config = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": "mindie-service",
                "labels": {
                    "app": "mindie-server"
                }
            },
            "spec": {
                "selector": {
                    "app": "mindie-server"
                },
                "ports": [
                    {
                        "name": "http",
                        "port": 1025,
                        "targetPort": 1025,
                        "nodePort": 31015,
                        "protocol": "TCP"
                    }
                ],
                "type": "NodePort",
                "sessionAffinity": "None"
            }
        }
        with open(self.yaml_path, 'w') as file:
            yaml.dump(service_config, file, default_flow_style=False)
        with open(self.config_single_pd_path, 'w') as fout:
            json.dump(pd_data, fout)
        self.config_single_bak_path = self.test_dir / "config_bak.json"
        self.config_single_pd_bak_path = self.test_dir / "ms_bak_controller.json"


    def tearDown(self):
        # 清理临时目录
        shutil.rmtree(self.test_dir)
        os.unlink(self.temp_file.name)

    def test_set_config_dict(self):
        origin_config = {"a": {"b": {"c": 3}}}
        DisaggregationSimulator.set_config(origin_config, "a.b.c", 4)
        assert origin_config["a"]["b"]["c"] == 4

    def test_set_config_list(self):
        origin_config = {"a": {"b": [{"c": 3}]}}
        DisaggregationSimulator.set_config(origin_config, "a.b.0.c", 4)
        assert origin_config["a"]["b"][0]["c"] == 4

    def test_set_config_new_key(self):
        origin_config = {"a": {"b": [{"c": 3}]}}
        DisaggregationSimulator.set_config(origin_config, "a.b.0.d", 4)
        assert origin_config["a"]["b"][0]["d"] == 4

    def test_set_config_add_dict_list_dict(self):
        origin_config = {"a": {"b": {"c": 3}}}
        DisaggregationSimulator.set_config(origin_config, "a.d.0.c", 4)
        assert origin_config["a"]["d"][0]["c"] == 4

    def test_set_config_add_dict(self):
        origin_config = {"a": {"b": [{"c": 3}]}}
        DisaggregationSimulator.set_config(origin_config, "a.b.1.c", 4)
        assert origin_config["a"]["b"][1]["c"] == 4

    def test_set_config_add_dict_list_dict_dict(self):
        origin_config = {"a": {"b": [{"c": 3}]}}
        DisaggregationSimulator.set_config(origin_config, "a.d.0.c.e", 4)
        assert origin_config["a"]["d"][0]["c"]["e"] == 4

    @patch('msserviceprofiler.modelevalstate.optimizer.simulator.logger')
    def test_is_int(self, mock_logger):
        # 测试is_int方法
        self.assertTrue(DisaggregationSimulator.is_int('123'))
        self.assertFalse(DisaggregationSimulator.is_int('abc'))

    @patch('msserviceprofiler.modelevalstate.optimizer.simulator.subprocess')
    @patch('msserviceprofiler.modelevalstate.optimizer.simulator.logger')
    def test_prepare_before_start_server(self, mock_logger, mock_subprocess):
        GlobalConfig.custom_return = True
        # 测试prepare_before_start_server方法
        mindie_config = KubectlConfig()
        simulator = DisaggregationSimulator(mindie_config)
        simulator.prepare_before_start_server()
        # 验证子进程是否正确运行
        mock_subprocess.run.assert_called()
        # 验证日志记录是否正确
        mock_logger.debug.assert_called()
        GlobalConfig.reset()

    @patch('msserviceprofiler.modelevalstate.optimizer.simulator.logger')
    def test_backup(self, mock_logger):
        # 测试backup方法
        mindie_config = KubectlConfig()
        bak_path = Path('/path/to/bak')
        simulator = DisaggregationSimulator(mindie_config, bak_path)
        simulator.backup()
        # 验证日志记录是否正确
        mock_logger.debug.assert_called()

    @patch('msserviceprofiler.modelevalstate.optimizer.simulator.logger')
    def test_stop(self, mock_logger):
        # 测试stop方法
        mindie_config = KubectlConfig()
        simulator = DisaggregationSimulator(mindie_config)
        simulator.stop()
        # 验证日志记录是否正确
        mock_logger.debug.assert_called()

    @patch('requests.post') 
    def test_curl_success(self, mock_post):
        # Arrange
        mindie_config = KubectlConfig()
        mindie_config.kubectl_single_path = self.kubectl_single_path
        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        test_class = DisaggregationSimulator(mindie_config) 

        # Act
        result = test_class.test_curl()

        # Assert
        self.assertTrue(result)
        mock_post.assert_called_once()

    @patch('requests.post') 
    def test_curl_failure(self, mock_post):
        # Arrange
        mindie_config = KubectlConfig()
        mindie_config.kubectl_single_path = self.kubectl_single_path
        mock_response = Mock()
        mock_response.status_code = 400
        mock_post.return_value = mock_response
        test_class = DisaggregationSimulator(mindie_config)  

        # Act
        result = test_class.test_curl()

        # Assert
        self.assertFalse(result)
        mock_post.assert_called_once()

    @patch('requests.post')  
    def test_curl_exception(self, mock_post):
        # Arrange
        mindie_config = KubectlConfig()
        mindie_config.kubectl_single_path = self.kubectl_single_path
        mock_post.side_effect = requests.exceptions.RequestException
        test_class = DisaggregationSimulator(mindie_config)  

        # Act
        result = test_class.test_curl()

        # Assert
        self.assertFalse(result)
        mock_post.assert_called_once()

    def test_update_config(self):
        # Arrange
        mindie_config = KubectlConfig()
        mindie_config.config_single_path = self.config_single_path
        mindie_config.config_single_pd_path = self.config_single_pd_path
        simulator = DisaggregationSimulator(mindie_config)
        
        # 创建测试参数
        params = [
            OptimizerConfigField(config_position="BackendConfig.ModelDeployConfig.maxSeqLen", value=4096),
            OptimizerConfigField(config_position="default_p_rate", value=2)
        ]
        
        # Act
        simulator.update_config(params)
        
        # Assert
        with open(self.config_single_path, 'r') as f:
            config_data = json.load(f)
            self.assertEqual(config_data["BackendConfig"]["ModelDeployConfig"]["maxSeqLen"], 4096)
        
        with open(self.config_single_pd_path, 'r') as f:
            pd_config_data = json.load(f)
            self.assertEqual(pd_config_data["default_p_rate"], 2)

    @patch('msserviceprofiler.modelevalstate.optimizer.simulator.DisaggregationSimulator.test_curl')
    @patch('msserviceprofiler.msguard.security.io.open_s')
    def test_check_success(self, mock_open, mock_test_curl):
        # Arrange
        GlobalConfig.custom_return = True
        mindie_config = KubectlConfig()
        simulator = DisaggregationSimulator(mindie_config)
        simulator.run_log = self.temp_file.name
        simulator.mindie_log_offset = 0
        
        # 模拟文件读取返回包含成功信息的内容
        mock_file = Mock()
        mock_file.read.return_value = "MindIE-MS coordinator is ready!!!"
        mock_file.tell.return_value = 100
        mock_open.return_value.__enter__.return_value = mock_file
        
        # 模拟test_curl返回True
        mock_test_curl.return_value = True
        
        # Act
        result = simulator.check_success()
        
        # Assert
        self.assertTrue(result)
        mock_test_curl.assert_called_once()
        GlobalConfig.reset()

    @patch('msserviceprofiler.modelevalstate.optimizer.simulator.DisaggregationSimulator.update_config')
    @patch('msserviceprofiler.modelevalstate.optimizer.simulator.DisaggregationSimulator.start_server')
    @patch('msserviceprofiler.modelevalstate.optimizer.simulator.logger')
    def test_run(self, mock_logger, mock_start_server, mock_update_config):
        # Arrange
        mindie_config = KubectlConfig()
        simulator = DisaggregationSimulator(mindie_config)
        
        # 创建测试参数
        params = [OptimizerConfigField(config_position="BackendConfig.ModelDeployConfig.maxSeqLen", value=4096)]
        
        # Act
        simulator.run(params)
        
        # Assert
        mock_logger.info.assert_called_once()
        mock_update_config.assert_called_once_with(params)
        mock_start_server.assert_called_once_with(params)