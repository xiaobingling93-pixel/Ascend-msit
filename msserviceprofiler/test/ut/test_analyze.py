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
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
from argparse import Namespace
import tempfile
import pytest

from ms_service_profiler.exporters.factory import ExporterFactory
from ms_service_profiler_ext.exporters.exporter_summary import ExporterSummary
from ms_service_profiler_ext.analyze import main, add_summary_exporter
from test.st.analyze.test_analyze_cmd_ms_service_profiler import check_csv_content


class TestMainFunction:
    ST_DATA_PATH = os.getenv("MS_SERVICE_PROFILER",
                             "/data/ms_service_profiler")
    REAL_INPUT_PATH = os.path.join(ST_DATA_PATH, "input/analyze/0211-1226")
    REQUEST_CSV = "request_summary.csv"
    BATCH_CSV = "batch_summary.csv"
    SERVICE_CSV = "service_summary.csv"

    @pytest.fixture(autouse=True)
    def mock_dependencies(self, mocker):
        mock_args = Namespace(
            input_path='/fake/input',
            output_path='/fake/output',
            log_level='info'
        )
        mocker.patch('argparse.ArgumentParser.parse_args', return_value=mock_args)
        mocker.patch('ms_service_profiler.utils.log.set_log_level')
        mocker.patch('ms_service_profiler.parse.preprocess_prof_folders')
        mocker.patch('ms_service_profiler.exporters.factory.ExporterFactory.create_exporters', return_value=[])
        mocker.patch.object(Path, 'mkdir')
        mocker.patch('ms_service_profiler.exporters.utils.create_sqlite_db')
        mocker.patch('os.path.exists', return_value=True)
        mocker.patch('ms_service_profiler.parse.find_file_in_dir', return_value=True)
        return mock_args

    @staticmethod
    def test_add_summary_exporter_decorator(mocker):
        mock_initialize = mocker.patch.object(ExporterSummary, 'initialize')
        original_create_exporters = MagicMock(return_value=['exporter1', 'exporter2'])
        wrapped_func = add_summary_exporter(original_create_exporters)

        args = Namespace(output_path='/fake/output')
        exporters = wrapped_func(args)

        assert len(exporters) == 3
        assert isinstance(exporters[-1], ExporterSummary)
        mock_initialize.assert_called_once_with(args)

    @staticmethod
    def test_command_line_interface(mocker):
        mock_main = mocker.patch('ms_service_profiler_ext.analyze.main')
        mocker.patch('sys.argv', ['script_name', '--input-path', '/fake/input'])

        import ms_service_profiler_ext.analyze
        ms_service_profiler_ext.analyze.main()
        mock_main.assert_called_once()

    @staticmethod
    def test_invalid_input_path(mocker):
        mocker.patch(
            'argparse.ArgumentParser.parse_args',
            side_effect=ValueError("Invalid path: '/invalid/path'")
        )
        with pytest.raises(ValueError, match=r"Invalid path.*"):
            main()

    @staticmethod
    def test_main_with_real_data(mocker):
        with tempfile.TemporaryDirectory() as tmp_output:
            real_args = Namespace(
                input_path=TestMainFunction.REAL_INPUT_PATH,
                output_path=tmp_output,
                log_level="info"
            )

            print(real_args)

            mocker.patch('argparse.ArgumentParser.parse_args', return_value=real_args)

            main()

            output_path = Path(tmp_output)
            assert (output_path / "profiler.db").is_file(), "数据库文件未生成"

            csv_checks = [
                (TestMainFunction.REQUEST_CSV, ['Metric', 'Average', 'Max', 'Min', 'P50', 'P90', 'P99'],
                 ['Average', 'Max', 'Min', 'P50', 'P90', 'P99']),
                (TestMainFunction.BATCH_CSV, ['Metric', 'Average', 'Max', 'Min', 'P50', 'P90', 'P99'],
                 ['Average', 'Max', 'Min', 'P50', 'P90', 'P99']),
                (TestMainFunction.SERVICE_CSV, ['Metric', 'Value'], ['Value'])
            ]

            for csv_file, expected_columns, numeric_columns in csv_checks:
                assert check_csv_content(
                    tmp_output,
                    csv_file,
                    expected_columns,
                    numeric_columns
                ), f"{csv_file} 内容校验失败"