import unittest

from components.utils.check.string_checker import StringChecker


class TestStringChecker(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.pass_msg = "pass"

    def setUp(self):
        self.sc = StringChecker()

    def test_not_str(self):
        err_msg = "is not a string"

        INVALID_STR = [1, 2.5, -3j, b'abc', (1,), [1, 2], {1, 2, 3}]
        for path in INVALID_STR:
            with self.subTest(path=path):
                res_msg = str(StringChecker().is_str().check(path))
                self.assertRegex(res_msg, err_msg)
    
    def test_str(self):
        path = "abc"
        res_msg = str(StringChecker().is_str().check(path))
        self.assertRegex(res_msg, self.pass_msg)
    
    def test_name_too_long(self):
        err_msg = "File name too long"

        INVALID_STR = ["s/" * 2048, 's' * 256]
        for path in INVALID_STR:
            with self.subTest(path=path):
                res_msg = str(StringChecker().is_file_name_too_long().check(path))
                self.assertEqual(res_msg, err_msg)
    
    def test_name_not_too_long(self):
        VALID_STR = ["a", "b", "c", "ab", "ac", "bc", "abc"]
        for path in VALID_STR:
            with self.subTest(path=path):
                res_msg = str(StringChecker().is_file_name_too_long().check(path))
                self.assertEqual(res_msg, self.pass_msg)

    def test_str_not_safe(self):
        err_msg = "String parameter contains invalid characters"

        INVALID_STR = ['&', '+', '@', '#', '$']
        for path in INVALID_STR:
            with self.subTest(path=path):
                res_msg = str(StringChecker().is_str_safe().check(path))
                self.assertEqual(res_msg, err_msg)

    def test_str_safe(self):
        VALID_STR = ['a', 'b', 'a_b', 'c-d', 'b=d', 'rm -rf /', 'echo xxx > /dev/null']
        for path in VALID_STR:
            with self.subTest(path=path):
                res_msg = str(StringChecker().is_str_safe().check(path))
                self.assertEqual(res_msg, self.pass_msg)

    def test_str_not_valid_bool(self):
        err_msg = "Boolean value expected 'yes', 'y', 'Y', 'YES', 'true', 't', 'TRUE', 'True', '1' for true"

        INVALID_STR = ['n', 'no', '\n', '\t', 'k']
        for path in INVALID_STR:
            with self.subTest(path=path):
                res_msg = str(StringChecker().is_str_valid_bool().check(path))
                self.assertEqual(res_msg, err_msg)

    def test_str_valid_bool(self):
        VALID_STR = ['y', 'yes', 't', 'true', 'true', '1']
        for path in VALID_STR:
            with self.subTest(path=path):
                res_msg = str(StringChecker().is_str_valid_bool().check(path))
                self.assertEqual(res_msg, self.pass_msg)

    def test_str_not_VALID_STR(self):
        err_msg = "Input path contains invalid characters"

        INVALID_STR = ['>', '>>', ' xxx@xxx', '1+1=2', 'rm -rf /']
        for path in INVALID_STR:
            with self.subTest(path=path):
                res_msg = str(StringChecker().is_str_valid_path().check(path))
                self.assertEqual(res_msg, err_msg)
    
    def test_str_VALID_STR(self):
        VALID_STR = ['a', 'b', '1_b', 'c-d']
        for path in VALID_STR:
            with self.subTest(path=path):
                res_msg = str(StringChecker().is_str_valid_path().check(path))
                self.assertEqual(res_msg, self.pass_msg)
    
    def test_str_not_valid_ids(self):
        err_msg = "dym range string"

        INVALID_STR = ['>', '>>', ' xxx@xxx', '1+1=2', 'rm -rf /', '1_2 ', '1_2, 123']
        for path in INVALID_STR:
            with self.subTest(path=path):
                res_msg = str(StringChecker().is_str_valid_ids().check(path))
                self.assertRegex(res_msg, err_msg)

    def test_str_valid_ids(self):
        VALID_STR = ['1_2', '2_3,4_5', '4_5,6_7,796_12321', '123']
        for path in VALID_STR:
            with self.subTest(path=path):
                res_msg = str(StringChecker().is_str_valid_ids().check(path))
                self.assertRegex(res_msg, self.pass_msg)
    
    def test_str_has_invalid_char(self):
        err_msg = "Input string contains invalid chars"

        INVALID_STR = ['&', '\n', '\f', '\u007F', '@', ';', '#']
        for path in INVALID_STR:
            with self.subTest(path=path):
                res_msg = str(StringChecker().str_has_no_invalid_char().check(path))
                self.assertRegex(res_msg, err_msg)

    def test_str_has_no_invalid_char(self):
        VALID_STR = ["a", "b", "c", "ab", "ac", "bc", "abc"]
        for path in VALID_STR:
            with self.subTest(path=path):
                res_msg = str(StringChecker().is_file_name_too_long().check(path))
                self.assertEqual(res_msg, self.pass_msg)
