import unittest
from utility.EqualSumCombinations import EqualSumCombinations

class testEqualSumCombinations(unittest.TestCase):
    def test_get_combinations(self):
        x = EqualSumCombinations(2, 2)
        self.assertTrue(len(x.get_combinations(2, 2)) == 3)
        x = EqualSumCombinations(2, 3)
        self.assertTrue(len(x.get_combinations(2, 3)) == 6)