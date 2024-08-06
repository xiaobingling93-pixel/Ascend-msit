import sys
import os
from unittest import TestCase
from unittest.mock import patch

sys.path.append('/root/workspace/msit/msit/')
sys.path.append('/root/workspace/msit/msit/components/llm')
from msit_llm import Synthesizer


class TestSynthezier(TestCase):

    def setUp(self) -> None:
        self.synthezier = Synthesizer()

    def test_synthezier_with_no_args_constructor(self):
        self.assertTrue(all(value.size == 0 for value in self.synthezier._info.values()))

    def test_synthezier_with_multiple_entries_args_constructor(self):
        import pandas as pd
        import numpy as np

        self.synthezier = Synthesizer(
            queries=['English', 'Math'],
            input_token_ids=([1, 2, 3], [4, 5, 6]),
            output_token_ids=pd.Series([[1], [2, 3]]),
            passed=np.array(['True', 'False'])
        )

        self.assertTrue(all(value.size for value in self.synthezier._info.values()))
        
    def test_synthezier_with_single_entry_args_constructor(self):
        self.synthezier = Synthesizer(
            queries='Here',
            input_token_ids=2.5,
            output_token_ids=2+3j,
            passed=True
        )

        self.assertTrue(all(value.size for value in self.synthezier._info.values()))
    
    def test_synthezier_to_csv(self):
        self.synthezier.from_args(
            queries='Question 1',
            input_token_ids=[[1, 2, 3], [4, 5, 6]],
            output_token_ids=[[7, 8, 9, 10, 11, 12]],
            passed='Correct'
        )

        self.synthezier.to_csv()
        self.assertTrue(any(filename.startswith('msit_synthesizer') and filename.endswith('.csv') for filename in os.listdir()))

    def test_synthezier_to_csv_first(self):
        with self.assertLogs('msit_llm_logger', 'ERROR') as cm:
            self.assertRaises(RuntimeError, self.synthezier.to_csv)
            logger_output = cm.output
            self.assertEqual(len(logger_output), 2)
            self.assertRegex(logger_output[0], r'should be called first before')

    def test_synthezier_to_csv_pad(self):
        self.synthezier.from_args(
            queries='Question 1',
            input_token_ids=[[0, 1, 2, 3, 4, 5], 2, 3],
            output_token_ids=[[7, 8, 9, 10, 11, 12], 4, 5],
            passed=('Correct', 'Wrong')
        )

        self.synthezier.to_csv(errors='pad')
        for value in self.synthezier._info.values():
            self.assertTrue(value.size, 1)

    def test_synthezier_to_csv_strict(self):
        self.synthezier.from_args(
            queries='Question 1',
            input_token_ids=[[0, 1, 2, 3, 4, 5], 2, 3],
            output_token_ids=[[7, 8, 9, 10, 11, 12], 4, 5],
            passed=('Correct', 'Wrong')
        )

        self.assertRaises(ValueError, self.synthezier.to_csv, errors='strict')

    def test_synthezier_to_csv_other_value(self):
        self.synthezier.from_args(
            queries='Question 1',
            input_token_ids=[[0, 1, 2, 3, 4, 5], 2, 3],
            output_token_ids=[[7, 8, 9, 10, 11, 12], 4, 5],
            passed=('Correct', 'Wrong')
        )

        with self.assertLogs('msit_llm_logger', 'ERROR') as cm:
            self.assertRaises(ValueError, self.synthezier.to_csv, errors='qweqwe')
            logger_output = cm.output
            self.assertEqual(len(logger_output), 1)
            self.assertRegex(logger_output[0], r'Wrong value')

    def test_synthezier_from_cmd(self):
        with self.assertLogs('msit_llm_logger', 'INFO') as cm:
            self.synthezier.from_cmd('python3 -c "print(2)"')
            logger_output = cm.output
            self.assertEqual(len(logger_output), 1)
            self.assertRegex(logger_output[0], r'all the outs and errors are suppressed$')

    def test_synthezier_from_cmd_no_patcher_folder(self):
        with self.assertLogs('msit_llm_logger', 'ERROR') as cm:
            with patch('os.path.exists', return_value=False):
                self.assertRaises(FileNotFoundError, self.synthezier.from_cmd, 'echo -c asdsad')
                logger_output = cm.output
                self.assertEqual(len(logger_output), 1)
                self.assertRegex(logger_output[0], r'not found')

    def test_synthezier_from_not_found_cmd(self):
        with self.assertLogs('msit_llm_logger', 'ERROR') as cm:
            self.assertRaises(OSError, self.synthezier.from_cmd, 'asdasd aaa')
            logger_output = cm.output
            self.assertEqual(len(logger_output), 1)
            self.assertRegex(logger_output[0], r'command not found')

    def test_synthezier_from_invalid_cmd(self):
        with self.assertLogs('msit_llm_logger', 'ERROR') as cm:
            self.assertRaises(ValueError, self.synthezier.from_cmd, 'rm')
            logger_output = cm.output
            self.assertEqual(len(logger_output), 1)
            self.assertRegex(logger_output[0], r'Invalid command')
    
    def test_synthezier_from_error_cmd(self):
        with self.assertLogs('msit_llm_logger', 'ERROR') as cm:
            self.assertRaises(RuntimeError, self.synthezier.from_cmd, 'git aaa')
            logger_output = cm.output
            self.assertEqual(len(logger_output), 1)
            self.assertRegex(logger_output[0], r'Failed to run command')
    
    @classmethod
    def tearDownClass(cls) -> None:
        for filename in os.listdir():
            if filename.startswith('msit_') and filename.endswith('.csv'):
                os.remove(filename)

# 100%