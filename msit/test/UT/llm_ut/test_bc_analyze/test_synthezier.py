# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------
import os
from unittest import TestCase

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
            queries=["English", "Math"],
            input_token_ids=([1, 2, 3], [4, 5, 6]),
            output_token_ids=pd.Series([[1], [2, 3]]),
            passed=np.array(["True", "False"]),
        )

        self.assertTrue(all(value.size for value in self.synthezier._info.values()))

    def test_synthezier_with_single_entry_args_constructor(self):
        self.synthezier = Synthesizer(queries="Here", input_token_ids=2.5, output_token_ids=2 + 3j, passed=True)

        self.assertTrue(all(value.size for value in self.synthezier._info.values()))

    def test_synthezier_to_csv(self):
        self.synthezier.from_args(
            queries="Question 1",
            input_token_ids=[[1, 2, 3], [4, 5, 6]],
            output_token_ids=[[7, 8, 9, 10, 11, 12]],
            passed="Correct",
        )

        self.synthezier.to_csv()
        self.assertTrue(os.path.exists("msit_bad_case/synthesizer"))
        self.assertTrue(
            any(
                filename.endswith('.csv') 
                for filename in os.listdir('msit_bad_case/synthesizer')
            )
        )

    def test_synthezier_to_csv_pad(self):
        self.synthezier.from_args(
            queries="Question 1",
            input_token_ids=[[0, 1, 2, 3, 4, 5], 2, 3],
            output_token_ids=[[7, 8, 9, 10, 11, 12], 4, 5],
            passed=("Correct", "Wrong"),
        )

        self.synthezier.to_csv(errors="pad")
        for value in self.synthezier._info.values():
            self.assertTrue(value.size, 1)

    def test_synthezier_to_csv_strict(self):
        self.synthezier.from_args(
            queries="Question 1",
            input_token_ids=[[0, 1, 2, 3, 4, 5], 2, 3],
            output_token_ids=[[7, 8, 9, 10, 11, 12], 4, 5],
            passed=("Correct", "Wrong"),
        )

        self.assertRaises(ValueError, self.synthezier.to_csv, errors="strict")

    def test_synthezier_to_csv_other_value(self):
        self.synthezier.from_args(
            queries='Question 1',
            input_token_ids=[[0, 1, 2, 3, 4, 5], 2, 3],
            output_token_ids=[[7, 8, 9, 10, 11, 12], 4, 5],
            passed=('Correct', 'Wrong')
        )

        self.assertRaises(ValueError, self.synthezier.to_csv, errors='qweqwe')
    
    @classmethod
    def tearDownClass(cls) -> None:
        if os.path.exists("msit_bad_case"):
            import shutil
            shutil.rmtree('msit_bad_case')
