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
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from msserviceprofiler.modelevalstate.optimizer import analyze_profiler
from msserviceprofiler.modelevalstate.optimizer.analyze_profiler import analyze


class TestFindFirstSimulateCSV(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)
    
    def test_valid_directory_with_matching_files(self):
        """Test with directory containing matching simulate*.csv files"""
        # Create test files
        open(os.path.join(self.test_dir, "simulate1.csv"), 'a').close()
        open(os.path.join(self.test_dir, "simulate2.csv"), 'a').close()
        open(os.path.join(self.test_dir, "other.csv"), 'a').close()
        
        result = analyze_profiler.find_first_simulate_csv(self.test_dir)
        self.assertEqual(result, os.path.join(self.test_dir, "simulate1.csv"))
    
    def test_valid_directory_no_matching_files(self):
        """Test with directory containing no simulate*.csv files"""
        # Create non-matching files
        open(os.path.join(self.test_dir, "test1.csv"), 'a').close()
        open(os.path.join(self.test_dir, "data.csv"), 'a').close()
        
        with self.assertRaises(FileNotFoundError) as context:
            analyze_profiler.find_first_simulate_csv(self.test_dir)
        self.assertEqual(str(context.exception), 
                         "No CSV files starting with 'simulate' found in the directory.")
    
    def test_nonexistent_directory(self):
        """Test with non-existent directory"""
        nonexistent_path = os.path.join(self.test_dir, "nonexistent")
        with self.assertRaises(NotADirectoryError) as context:
            analyze_profiler.find_first_simulate_csv(nonexistent_path)
        self.assertEqual(str(context.exception), 
                         "The provided path is not a valid directory.")
    
    def test_file_instead_of_directory(self):
        """Test when path points to a file instead of directory"""
        file_path = os.path.join(self.test_dir, "testfile")
        open(file_path, 'a').close()
        
        with self.assertRaises(NotADirectoryError) as context:
            analyze_profiler.find_first_simulate_csv(file_path)
        self.assertEqual(str(context.exception), 
                         "The provided path is not a valid directory.")
    
    def test_empty_directory(self):
        """Test with empty directory"""
        with self.assertRaises(FileNotFoundError) as context:
            analyze_profiler.find_first_simulate_csv(self.test_dir)
        self.assertEqual(str(context.exception), 
                         "No CSV files starting with 'simulate' found in the directory.")
    
    def test_file_sorting(self):
        """Test that files are properly sorted"""
        # Create files in non-sorted order
        open(os.path.join(self.test_dir, "simulate10.csv"), 'a').close()
        open(os.path.join(self.test_dir, "simulate2.csv"), 'a').close()
        open(os.path.join(self.test_dir, "simulate1.csv"), 'a').close()
        
        result = analyze_profiler.find_first_simulate_csv(self.test_dir)
        self.assertEqual(result, os.path.join(self.test_dir, "simulate1.csv"))
