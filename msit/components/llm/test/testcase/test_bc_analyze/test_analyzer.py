import os
from unittest import TestCase
from unittest.mock import patch

import pandas as pd

from msit_llm import Analyzer, Synthesizer


class TestSynthezier(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.golden_csv_path = 'golden_csv_path.csv'
        cls.golden_dict = {
                'queries': ['How are you?', 'Hello', 'What is your name?', 'What time is it?'],
                'input_token_ids': [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16, 17], [18, 19, 20, 21, 22]],
                'output_token_ids': [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16, 17], [18, 19, 20, 21, 22]],
                'passed': ['Correct', 'Wrong', 'Correct', 'Correct']
            }
        golden_df = pd.DataFrame(cls.golden_dict)
        golden_df.to_csv(cls.golden_csv_path, encoding='utf-8', index=False)

        cls.test_csv_path = 'test_csv_path.csv'
        cls.test_dict = {
                'queries': ['How are you?', 'Hello', 'What is your name?', 'What time is it?'],
                'input_token_ids': [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16, 17], [18, 19, 20, 21, 22]],
                'output_token_ids': [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16, 17], [18, 19, 20, 21, 22]],
                'passed': ['Wrong', 'Correct', 'Wrong', 'Correct']
            }
        
        test_df = pd.DataFrame(cls.test_dict)
        test_df.to_csv(cls.test_csv_path, encoding='utf-8', index=False)
        cls.analyzer = Analyzer()

    def test_analyzer_from_csv(self):
        self.analyzer.from_csv(golden_csv_path=self.golden_csv_path, test_csv_path=self.test_csv_path)

        self.assertTrue(os.path.exists('msit_bad_case_analyze'))
        self.assertTrue(any(file_name.startswith('msit_bad_case_') and file_name.endswith('.csv') for file_name in os.listdir('msit_bad_case_analyze')))
    
    def test_analyzer_from_csv_arg_not_str(self):
        with self.assertLogs('msit_llm_logger', 'ERROR') as cm:
            self.assertRaises(ValueError, self.analyzer.from_csv, golden_csv_path=2, test_csv_path=self.test_csv_path)
            logger_output = cm.output
            self.assertEqual(len(logger_output), 1)
            self.assertRegex(logger_output[0], r'Invalid csv path, only path with suffix')

    def test_analyzer_from_csv_arg_not_exist(self):
        with self.assertLogs('msit_llm_logger', 'ERROR') as cm:
            self.assertRaises(OSError, self.analyzer.from_csv, golden_csv_path='aqewwqe.csv', test_csv_path=self.test_csv_path)
            logger_output = cm.output
            self.assertEqual(len(logger_output), 1)
            self.assertRegex(logger_output[0], r'aqewwqe.csv')
    
    def test_analyzer_from_csv_not_owner(self):
        with self.assertLogs('msit_llm_logger', 'ERROR') as cm:
            with patch('os.getuid', return_value=1000):
                self.assertRaises(PermissionError, self.analyzer.from_csv, golden_csv_path=self.golden_csv_path, test_csv_path=self.test_csv_path)
                logger_output = cm.output
                self.assertEqual(len(logger_output), 1)
                self.assertRegex(logger_output[0], r'should be used only by its owner or superuser')
    
    def test_analyzer_from_csv_other_writeable(self):
        with self.assertLogs('msit_llm_logger', 'ERROR') as cm:
            with patch('os.stat', return_value=os.stat_result([2] + [0] * 9)):
                self.assertRaises(PermissionError, self.analyzer.from_csv, golden_csv_path=self.golden_csv_path, test_csv_path=self.test_csv_path)
                logger_output = cm.output
                self.assertEqual(len(logger_output), 1)
                self.assertRegex(logger_output[0], r'should be used only by its owner or superuser')
    
    def test_analyzer_from_mixed(self):
        import time
        time.sleep(1) # The analyze is so fast that the time stamp does not change, leads to permssion error
        self.analyzer.from_mixed(golden=Synthesizer(**self.golden_dict), test=self.test_csv_path)
        time.sleep(1)
        self.analyzer.from_mixed(golden=self.golden_csv_path, test=Synthesizer(**self.test_dict))

        self.assertTrue(os.path.exists('msit_bad_case_analyze'))
        self.assertTrue(any(file_name.startswith('msit_bad_case_') and file_name.endswith('.csv') for file_name in os.listdir('msit_bad_case_analyze')))
        time.sleep(1)

    def test_analyzer_unmatched_df(self):
        test_dict = {
                'queries': ['How are you?', 'What is your name?', 'What time is it?', 'No!'],
                'input_token_ids': [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16, 17], [18, 19, 20, 21, 22]],
                'output_token_ids': [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16, 17], [18, 19, 20, 21, 22]],
                'passed': ['Wrong', 'Correct', 'Wrong', 'Correct']
            }
        
        test = Synthesizer(**test_dict)
        with self.assertLogs('msit_llm_logger', 'WARNING') as cm: 
            self.analyzer.from_mixed(golden=self.golden_csv_path, test=test)
            logger_output = cm.output
            self.assertEqual(len(logger_output), 1)
            self.assertRegex(logger_output[0], r'below is a partial display of these unmatched')

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.golden_csv_path)
        os.remove(cls.test_csv_path)

        if os.path.exists('msit_bad_case_analyze'):
            import shutil
            shutil.rmtree('msit_bad_case_analyze')

# 90%