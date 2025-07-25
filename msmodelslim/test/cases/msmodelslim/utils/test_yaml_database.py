# -*- coding: utf-8 -*-
# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
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
import pytest
from pathlib import Path
import yaml
import tempfile
import shutil
from msmodelslim.utils.yaml_database import YamlDatabase


class TestYamlDatabase:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """创建和清理测试用的临时目录"""
        # 创建临时目录结构
        self.test_dir = Path(tempfile.mkdtemp())
        self.valid_config_dir = self.test_dir / "valid_configs"
        self.valid_config_dir.mkdir()

        # 1. 保存原始 umask
        original_umask = os.umask(0)  # 临时设为 0 并获取原始值
        try:
            # 2. 设置目标 umask（0o026 对应权限 640）
            os.umask(0o026)

            # 创建有效的 YAML 文件
            config1 = self.valid_config_dir / "config1.yaml"
            config1.write_text("""
            name: Test Config 1
            version: 1.0
            settings:
              param1: value1
              param2: 42
            """)

            config2 = self.valid_config_dir / "config2.yaml"
            config2.write_text("""
            - item1
            - item2
            - item3
            """)

            # 创建空 YAML 文件
            empty_config = self.valid_config_dir / "empty.yaml"
            empty_config.write_text("")

            # 创建无效目录（文件）
            self.invalid_dir = self.test_dir / "invalid_dir"
            self.invalid_dir.write_text("This is not a directory")

        finally:
            # 4. 无论是否出错，都恢复原始 umask
            os.umask(original_umask)

        yield  # 测试执行
        
        # 清理
        shutil.rmtree(self.test_dir)


    def test_init_with_valid_directory(self):
        """测试使用有效目录初始化"""
        db = YamlDatabase(self.valid_config_dir)
        assert db.config_dir == self.valid_config_dir
        assert isinstance(db.config_by_name, dict)
        assert len(db.config_by_name) == 0


    def test_init_with_invalid_directory(self):
        """测试使用无效路径初始化"""
        with pytest.raises(ValueError) as excinfo:
            YamlDatabase(self.invalid_dir)
        assert "is not a directory" in str(excinfo.value)
        
        with pytest.raises(ValueError):
            YamlDatabase(Path("/nonexistent/path"))


    def test_load_config_success(self):
        """测试成功加载配置"""
        db = YamlDatabase(self.valid_config_dir)
        configs = list(db.load_config())
        
        # 应该加载了3个文件（包括空文件）
        assert len(configs) == 3


    def test_load_config_empty_directory(self):
        """测试加载空目录"""
        empty_dir = self.test_dir / "empty_dir"
        empty_dir.mkdir()
        
        db = YamlDatabase(empty_dir)
        configs = list(db.load_config())
        assert len(configs) == 0


    def test_load_config_permission_error(self, monkeypatch):
        """测试没有文件读取权限的情况"""
        db = YamlDatabase(self.valid_config_dir)
        
        def mock_open(*args, **kwargs):
            raise PermissionError("Permission denied")
        
        monkeypatch.setattr("builtins.open", mock_open)
        
        with pytest.raises(PermissionError):
            list(db.load_config())


    def test_load_config_with_non_yaml_files(self):
        """测试目录中包含非YAML文件的情况"""
        # 添加一个非YAML文件
        text_file = self.valid_config_dir / "notes.txt"
        text_file.write_text("This is not a YAML file")
        
        db = YamlDatabase(self.valid_config_dir)
        configs = list(db.load_config())
        
        # 应该只加载了3个YAML文件（不包括.txt）
        assert len(configs) == 3