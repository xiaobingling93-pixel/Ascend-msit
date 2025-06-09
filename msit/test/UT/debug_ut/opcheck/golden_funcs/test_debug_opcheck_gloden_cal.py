import unittest
import sys
from unittest.mock import patch, MagicMock
import numpy as np
from numpy.testing import assert_equal


class TestGoldenFuncsOperation(unittest.TestCase):

    def setUp(self):
        self.backup_modules = {}
        self.modules_to_mock = [
            'tensorflow',
            'tensorflow.compat',
            'tensorflow.compat.v1',
        ]
        for mod in self.modules_to_mock:
            if mod in sys.modules:
                self.backup_modules[mod] = sys.modules[mod]
            sys.modules[mod] = MagicMock()
        from msit_opcheck.golden_funcs import OP_DICT
        from msit_opcheck.operation_test import OperationTest
        self.OperationTest = OperationTest
        self.OP_DICT = OP_DICT
        
    def tearDown(self):
        for mod in self.modules_to_mock:
            if mod in self.backup_modules:
                sys.modules[mod] = self.backup_modules[mod]
            else:
                del sys.modules[mod] 
        for mod in list(sys.modules.keys()):
            if mod.startswith("msit_opcheck.golden_funcs"):
                del sys.modules[mod]

    def test_add_operation(self):
        with patch.object(self.OperationTest, "__init__", return_value=None):
            inputs = [np.array([1, 2, 3]), np.array([4, 5, 6])]
            add_op = self.OP_DICT["Add"]()
            add_op.op_param = {
                'output_desc': [{'dtype': 'DT_FLOAT'}]
            }
            result = add_op.golden_calc(inputs)[0]
            expected = np.array([5.0, 7.0, 9.0])
            assert_equal(result, expected)