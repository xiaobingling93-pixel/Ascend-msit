import unittest
import tempfile
import os

from msprechecker.core.validators.range import RangeValidator


class TestRangeValidator(unittest.TestCase):
    def setUp(self):
        self.validator = RangeValidator()

    def test_validate_with_none_actual_value(self):
        self.assertFalse(self.validator.validate(None, "[1,2]"))

    def test_validate_with_non_numeric_actual_value(self):
        self.assertFalse(self.validator.validate("string", "[1,2]"))

    def test_validate_with_invalid_expected_value_type(self):
        with self.assertRaises(ValueError):
            self.validator.validate(1.5, {"invalid": "type"})

    def test_validate_with_invalid_string_format(self):
        with self.assertRaises(ValueError):
            self.validator.validate(1.5, "1,2,3")

    def test_validate_with_invalid_brackets(self):
        with self.assertRaises(ValueError):
            self.validator.validate(1.5, "<1,2>")

    def test_validate_with_non_numeric_bounds(self):
        with self.assertRaises(ValueError):
            self.validator.validate(1.5, "[a,b]")

    def test_validate_with_reversed_bounds(self):
        with self.assertRaises(ValueError):
            self.validator.validate(1.5, "[2,1]")

    def test_validate_with_equal_bounds(self):
        with self.assertRaises(ValueError):
            self.validator.validate(1.5, "[1,1]")

    def test_validate_with_inclusive_bounds(self):
        self.assertTrue(self.validator.validate(1, "[1,2]"))
        self.assertTrue(self.validator.validate(2, "[1,2]"))
        self.assertFalse(self.validator.validate(0, "[1,2]"))
        self.assertFalse(self.validator.validate(3, "[1,2]"))

    def test_validate_with_exclusive_bounds(self):
        self.assertTrue(self.validator.validate(1.5, "(1,2)"))
        self.assertFalse(self.validator.validate(1, "(1,2)"))
        self.assertFalse(self.validator.validate(2, "(1,2)"))

    def test_validate_with_mixed_bounds(self):
        self.assertTrue(self.validator.validate(1, "[1,2)"))
        self.assertFalse(self.validator.validate(2, "[1,2)"))
        self.assertTrue(self.validator.validate(2, "(1,2]"))
        self.assertFalse(self.validator.validate(1, "(1,2]"))

    def test_validate_with_list_input(self):
        self.assertTrue(self.validator.validate(1.5, [1, 2]))
        self.assertTrue(self.validator.validate(1, [1, 2]))
        self.assertTrue(self.validator.validate(2, [1, 2]))
        self.assertFalse(self.validator.validate(0, [1, 2]))
        self.assertFalse(self.validator.validate(3, [1, 2]))

    def test_validate_with_invalid_list_length(self):
        with self.assertRaises(ValueError):
            self.validator.validate(1.5, [1, 2, 3])

    def test_temp_file_usage(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = os.path.join(temp_dir, "test.txt")
            with open(temp_file, "w") as f:
                f.write("test content")
            self.assertTrue(os.path.exists(temp_file))
        self.assertFalse(os.path.exists(temp_file))

    def test_parse_expected_value_str_private_method_not_called_directly(self):
        # This test ensures we're not calling private methods directly
        # We test through the public validate method instead
        self.assertTrue(self.validator.validate(1.5, "[1,2]"))

    def test_parse_expected_value_list_private_method_not_called_directly(self):
        # This test ensures we're not calling private methods directly
        # We test through the public validate method instead
        self.assertTrue(self.validator.validate(1.5, [1, 2]))
