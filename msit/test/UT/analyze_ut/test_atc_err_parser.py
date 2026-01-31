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
import unittest
import os

from typing import List

from model_evaluation.common.enum import AtcErr
from model_evaluation.parser import AtcErrParser


class TestAtcErrParser(unittest.TestCase):
    def setUp(self) -> None:
        err_file_map = {
            AtcErr.E19010: 'atc_err_e19010.txt',
            AtcErr.EZ9010: 'atc_err_ez9010.txt',
            AtcErr.EZ0501: 'atc_err_ez0501.txt',
            AtcErr.EZ3002: 'atc_err_ez3002.txt',
            AtcErr.EZ3003: 'atc_err_ez3003.txt',
            AtcErr.UNKNOWN: 'atc_err_unknown.txt',
        }
        self.errinfo_map = {}
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        err_path = os.path.join(cur_dir, '..', 'resource', 'analyze', 'dataset', 'atc_err')
        for errcode, err_file in err_file_map.items():
            errfile_path = os.path.join(err_path, err_file)
            if not os.path.isfile(errfile_path):
                continue
            with open(errfile_path) as f:
                lines: List[str] = f.readlines()
                errinfo = ''.join(lines)
                self.errinfo_map[errcode] = errinfo

    def test_parse_errcode_success_case(self):
        for errcode, errinfo in self.errinfo_map.items():
            res = AtcErrParser.parse_errcode(errinfo)
            self.assertEqual(res, errcode)

    def test_parse_err_e19010(self):
        errinfo = self.errinfo_map.get(AtcErr.E19010)
        err_ops = AtcErrParser.parse_unsupported_op(errinfo)
        self.assertIsInstance(err_ops, dict)

        real_err_ops = {
            'Conv_52': 'ai.onnx::11::Convx',
            'Conv_112': 'ai.onnx::11::Convx',
            'Conv_172': 'ai.onnx::11::Convx',
            'Conv_232': 'ai.onnx::11::Convx',
            'Conv_292': 'ai.onnx::11::Convx',
        }
        self.assertEqual(err_ops, real_err_ops)

    def test_parse_err_ez9010(self):
        errinfo = self.errinfo_map.get(AtcErr.EZ9010)
        err_ops = AtcErrParser.parse_unsupported_op(errinfo)
        self.assertIsInstance(err_ops, dict)

        real_err_ops = {
            'Conv_52': 'ai.onnx::11::Convx',
            'Conv_112': 'ai.onnx::11::Convx',
            'Conv_172': 'ai.onnx::11::Convx',
            'Conv_232': 'ai.onnx::11::Convx',
            'Conv_292': 'ai.onnx::11::Convx',
        }
        self.assertEqual(err_ops, real_err_ops)

    def test_parse_err_ez3003(self):
        errinfo = self.errinfo_map.get(AtcErr.EZ3003)
        err_ops = AtcErrParser.parse_unsupported_op(errinfo)
        self.assertIsInstance(err_ops, dict)

        real_err_ops = {
            'Conv_52': 'Conv',
            'Conv_112': 'Conv',
            'Conv_172': 'Conv',
            'Conv_232': 'Conv',
            'Conv_292': 'Conv',
        }
        self.assertEqual(err_ops, real_err_ops)

    def test_parse_err_ez3002(self):
        errinfo = self.errinfo_map.get(AtcErr.EZ3002)
        err_ops = AtcErrParser.parse_unsupported_op(errinfo)
        self.assertIsInstance(err_ops, set)

        real_err_ops = {'Convx'}
        self.assertEqual(err_ops, real_err_ops)

    def test_parse_err_ez0501(self):
        errinfo = self.errinfo_map.get(AtcErr.EZ0501)
        err_ops = AtcErrParser.parse_unsupported_op(errinfo)
        self.assertIsInstance(err_ops, dict)

        real_err_ops = {
            'Conv_52': 'Conv',
            'Conv_112': 'Conv',
            'Conv_172': 'Conv',
            'Conv_232': 'Conv',
            'Conv_292': 'Conv',
        }
        self.assertEqual(err_ops, real_err_ops)

    def test_parse_err_unknown(self):
        errinfo = self.errinfo_map.get(AtcErr.UNKNOWN)
        err_ops = AtcErrParser.parse_unsupported_op(errinfo)
        self.assertEqual(err_ops, {})


if __name__ == "__main__":
    unittest.main()
