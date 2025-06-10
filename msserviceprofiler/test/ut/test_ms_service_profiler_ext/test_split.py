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

import unittest

from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pytest

from ms_service_profiler_ext.split import add_exporters, main
from ms_service_profiler_ext.exporters.exporter_prefill import ExporterPrefill
from ms_service_profiler_ext.exporters.exporter_decode import ExporterDecode


class TestSplitFuctions(unittest.TestCase):
    mock_args = None
    mocker = None

    @property
    def mock_args(self):
        return self.__class__.mock_args

    @mock_args.setter
    def mock_args(self, value):
        self.__class__.mock_args = value

    @property
    def mocker(self):
        return self.__class__.mocker

    @mocker.setter
    def mocker(self, value):
        self.__class__.mocker = value

    def test_add_exporters_with_prefill(self):
        args = Namespace(prefill_batch_size=4, decode_batch_size=0, prefill_rid="-1", decode_rid="-1")
        exporters = add_exporters(args)

        self.assertEqual(len(exporters), 1)
        self.assertIsInstance(exporters[0], ExporterPrefill)

    def test_add_exporters_with_decode(self):
        args = Namespace(prefill_batch_size=0, decode_batch_size=10, prefill_rid="-1", decode_rid="-1")
        exporters = add_exporters(args)

        self.assertEqual(len(exporters), 1)
        self.assertIsInstance(exporters[0], ExporterDecode)

    def test_add_exporters_with_both(self):
        args = Namespace(prefill_batch_size=4, decode_batch_size=10, prefill_rid="-1", decode_rid="-1")
        exporters = add_exporters(args)

        self.assertEqual(len(exporters), 2)
        self.assertIsInstance(exporters[0], ExporterPrefill)
        self.assertIsInstance(exporters[1], ExporterDecode)

    def test_main(self):
        main()

    @pytest.fixture(autouse=True)
    def _inject_mocker(self, mocker):
        self.mocker = mocker
        self.mock_args = Namespace(input_path="/fake/input", output_path="/fake/output", log_level="info", format="csv")

        # 2. 配置全局mock
        mocker.patch("argparse.ArgumentParser.parse_args", return_value=self.mock_args)
        mocker.patch("ms_service_profiler.utils.log.set_log_level")
        mocker.patch("ms_service_profiler.parse.preprocess_prof_folders")
        mocker.patch("ms_service_profiler_ext.split.add_exporters", return_value=[ExporterPrefill, ExporterDecode])
        mocker.patch.object(Path, "mkdir")
        mocker.patch("os.path.exists", return_value=True)
        mocker.patch("ms_service_profiler.parse.find_file_in_dir", return_value=True)
        mocker.patch("os.makedirs")

        yield
        self.mocker = None
        self.mock_args = None
