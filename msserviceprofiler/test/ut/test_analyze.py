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

import logging
import sqlite3
from argparse import Namespace
from pathlib import Path
import pytest
import unittest
from unittest.mock import MagicMock, ANY

from ms_service_profiler.exporters.factory import ExporterFactory
from ms_service_profiler.utils.log import set_logger
from ms_service_profiler.parse import parse, preprocess_prof_folders, find_file_in_dir
from ms_service_profiler.exporters.utils import check_input_path_valid, check_output_path_valid, create_sqlite_db
from ms_service_profiler_ext.exporters.exporter_summary import ExporterSummary
from ms_service_profiler_ext.analyze import main, add_summary_exporter, set_log_level


class TestMainFunction:
    @pytest.fixture(autouse=True)
    def mock_dependencies(self, mocker):
        self.mock_args = Namespace(
            input_path='/fake/input',
            output_path='/fake/output',
            log_level='info'
        )
        mocker.patch('argparse.ArgumentParser.parse_args', return_value=self.mock_args)

        mocker.patch('ms_service_profiler.utils.log.set_log_level')
        mocker.patch('ms_service_profiler.parse.preprocess_prof_folders')
        mocker.patch('ms_service_profiler.exporters.factory.ExporterFactory.create_exporters', return_value=[])
        mocker.patch.object(Path, 'mkdir')
        mocker.patch('ms_service_profiler.exporters.utils.create_sqlite_db')

        mocker.patch('os.path.exists', return_value=True)
        mocker.patch('ms_service_profiler.parse.find_file_in_dir', return_value=True)

    def test_add_summary_exporter_decorator(self, mocker):
        mock_initialize = mocker.patch.object(ExporterSummary, 'initialize')
        original_create_exporters = MagicMock(return_value=['exporter1', 'exporter2'])
        wrapped_func = add_summary_exporter(original_create_exporters)

        args = Namespace(output_path='/fake/output')
        exporters = wrapped_func(args)

        assert len(exporters) == 3
        assert isinstance(exporters[-1], ExporterSummary)
        mock_initialize.assert_called_once_with(args)

    def test_command_line_interface(self, mocker):
        mock_main = mocker.patch('ms_service_profiler_ext.analyze.main')
        mocker.patch('sys.argv', ['script_name', '--input-path', '/fake/input'])

        import ms_service_profiler_ext.analyze
        ms_service_profiler_ext.analyze.main()
        mock_main.assert_called_once()

    def test_invalid_input_path(self, mocker):
        mocker.patch(
            'argparse.ArgumentParser.parse_args',
            side_effect=ValueError("Invalid path: '/invalid/path'")
        )
        with pytest.raises(ValueError, match=r"Invalid path.*"):
            main()