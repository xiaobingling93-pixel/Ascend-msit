from unittest import TestCase

from msit_llm import _RandomNameSequence


class TestUtils(TestCase):

    def test_random_name_with_timestamp(self):
        self.namer = _RandomNameSequence()
        
        for i, namer in zip(range(8), self.namer):
            with self.subTest(round=f'{i}th'):
                self.assertRegex(namer, r'[a-z0-9_]{8,}[_0-9]+')
        

    def test_random_name_with_no_timestamp(self):
        self.namer = _RandomNameSequence(False)
        
        for i, namer in zip(range(8), self.namer):
            with self.subTest(round=f'{i}th'):
                self.assertRegex(namer,  r'[a-z0-9_]{8,}')


if __name__ == "__main__":
    import unittest
    unittest.main()