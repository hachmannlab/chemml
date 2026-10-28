from chemml.chem.magpie_python.utility.EqualSumCombinations import EqualSumCombinations

class TestEqualSumCombinations:
    def test_get_combinations(self):
        x = EqualSumCombinations(2, 2)
        assert len(x.get_combinations(2, 2)) == 3
        x = EqualSumCombinations(2, 3)
        assert len(x.get_combinations(2, 3)) == 6