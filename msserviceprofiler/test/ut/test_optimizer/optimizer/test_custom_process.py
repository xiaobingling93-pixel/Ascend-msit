# -*- coding: utf-8 -*-
# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

from msserviceprofiler.modelevalstate.config.config import (
    CUSTOM_OUTPUT,
    MODEL_EVAL_STATE_CONFIG_PATH,
    OptimizerConfigField
)
from msserviceprofiler.modelevalstate.optimizer.custom_process import CustomProcess, tempfile, os


def test_before_run_no_run_params(monkeypatch):
    # 模拟 tempfile.mkstemp
    monkeypatch.setattr(tempfile, "mkstemp", lambda prefix="": (1234, 'tempfile'))
    # 模拟 os.environ
    monkeypatch.setattr(os, "environ", {})
    process = CustomProcess()
    process.before_run()

    # 验证属性设置
    assert process.run_log_fp == 1234
    assert process.run_log == 'tempfile'
    assert process.run_log_offset == 0


def test_before_run_with_run_params():
    process = CustomProcess()
    process.command = ["benchmark", "$CONCURRENCY", "$REQUESTRATE"]
    run_params = (
        OptimizerConfigField(name="CONCURRENCY", config_position="env", min=10, max=1000, dtype="int", value=10),
        OptimizerConfigField(name="REQUESTRATE", config_position="env", min=0.1, max=0.7, value=0.3, dtype="float"),
    )
    process.before_run(run_params)
    assert process.command == ["benchmark", "10", "0.3"]


def test_before_run_env_var_already_set(monkeypatch):
    # 模拟 os.environ
    monkeypatch.setattr(os, "environ", {CUSTOM_OUTPUT: "/result",
                                        MODEL_EVAL_STATE_CONFIG_PATH: "config.toml"})

    process = CustomProcess()
    process.before_run()

    # 验证 tempfile.mkstemp 被调用
    assert os.environ[CUSTOM_OUTPUT] == "/result"
    assert os.environ[MODEL_EVAL_STATE_CONFIG_PATH] == "config.toml"


def test_check_success_process_still_running(tmpdir):
    # 模拟子进程仍在运行
    custom_process = CustomProcess()
    custom_process.run_log = Path(tmpdir).joinpath("run_log")
    custom_process.run_log_offset = 0
    with open(custom_process.run_log, "w") as f:
        f.write("test")
    custom_process.process = Mock()
    custom_process.process.poll.return_value = None
    custom_process.print_log = True
    result = custom_process.check_success()

    assert result is False


def test_check_success_process_succeeded(tmpdir):
    # 模拟子进程成功完成
    custom_process = CustomProcess()
    custom_process.run_log = Path(tmpdir).joinpath("run_log")
    custom_process.run_log_offset = 0
    with open(custom_process.run_log, "w") as f:
        f.write("test")
    custom_process.process = Mock()
    custom_process.process.poll.return_value = 0
    custom_process.print_log = True
    result = custom_process.check_success()

    assert result is True


def test_check_success_process_failed(tmpdir):
    # 模拟子进程失败
    custom_process = CustomProcess()
    custom_process.run_log = Path(tmpdir).joinpath("run_log")
    custom_process.run_log_offset = 0
    with open(custom_process.run_log, "w") as f:
        f.write("test")
    custom_process.process = Mock()
    custom_process.process.poll.return_value = 1
    custom_process.print_log = True
    with pytest.raises(subprocess.SubprocessError) as e:
        custom_process.check_success()
