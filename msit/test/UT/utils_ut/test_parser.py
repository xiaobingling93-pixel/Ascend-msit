# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

from unittest.mock import MagicMock, patch

import pytest

from components.utils.parser import BaseCommand, LazyEntryPointCommand, AitCommand


@pytest.fixture
def mock_entry_point():
    entry_point = MagicMock()
    entry_point.name = "mock_entry:Mock help"
    entry_point.load.return_value = BaseCommand("test", "Test command")
    return entry_point


def test_base_command_register_parser_given_parser_when_valid_then_pass():
    # Mock argparse.ArgumentParser
    parser = MagicMock(spec=argparse.ArgumentParser)
    command = BaseCommand("test_command", "Test command description")

    # Mock add_arguments method
    command.add_arguments = MagicMock()

    command.register_parser(parser)

    command.add_arguments.assert_called_once()
    parser.set_defaults.assert_called_once()


def test_base_command_register_parser_given_parser_when_children_then_subparsers_created():
    parser = MagicMock(spec=argparse.ArgumentParser)
    child_command = BaseCommand("child_command", "Child command description")
    command = BaseCommand("test_command", "Test command description", children=[child_command])

    command.register_parser(parser)

    # Check subparsers are created
    parser.add_subparsers.assert_called_once()


def test_base_command_handle_given_parser_when_valid_then_pass():
    parser = MagicMock(spec=argparse.ArgumentParser)
    command = BaseCommand("test_command", "Test command description")

    command.register_parser(parser)
    command.handle(None)

    parser.print_help.assert_called_once()


def test_lazy_entry_point_command_build_lazy_tasks_given_valid_entry_points_then_tasks_created(mock_entry_point):
    # Mock get_entry_points to return a list with one mock entry point
    with patch('components.utils.parser.get_entry_points', return_value=[mock_entry_point]):
        tasks = LazyEntryPointCommand.build_lazy_tasks("mock_entry")

    assert len(tasks) > 0
    assert isinstance(tasks[0], LazyEntryPointCommand)


def test_lazy_entry_point_command_register_parser_given_invalid_child_then_raise_value_error():
    parser = MagicMock(spec=argparse.ArgumentParser)
    with pytest.raises(ValueError):
        command = BaseCommand("test_command", "Test command description", children=[123])  # Invalid child