from unittest import TestCase

from msit_llm.bc_analyze.utils import RandomNameSequence


class TestUtils(TestCase):

    def setUp(self) -> None:
        self.namer = RandomNameSequence()

    def test_random_name(self):    
        for i, namer in zip(range(8), self.namer):
            with self.subTest(round=f'{i}th'):
                self.assertRegex(namer, r'[a-z0-9_]{8,}[_0-9]+')
