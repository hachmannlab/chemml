import numpy as np
import numpy.testing as np_tst
from chemml.chem.magpie_python.vassal.util.VectorCombinationComputer import VectorCombinationComputer
import math

class TestVectorCombinationComputer:
    def setup_method(self):
        self.input = list(np.identity(3))

    def teardown_method(self):
        self.input = None

    def test_combination(self):
        computer = VectorCombinationComputer(self.input, 1)

        v = np.array([1, 0, 0])
        np_tst.assert_array_almost_equal(v, computer.compute_vector(v))
        v = np.array([0, 1, 0])
        np_tst.assert_array_almost_equal(v, computer.compute_vector(v))
        v = np.array([0, 0, 1])
        np_tst.assert_array_almost_equal(v, computer.compute_vector(v))
        v = np.array([-2, 5, 3])
        np_tst.assert_array_almost_equal(v, computer.compute_vector(v))

    def test_collection(self):
        computer = VectorCombinationComputer(self.input, 1)
        assert 7 == len(computer.get_vectors())

        computer = VectorCombinationComputer(self.input, math.sqrt(2) * 1.001)
        assert 1 + 6 + 12 == len(computer.get_vectors())

        computer = VectorCombinationComputer(self.input, math.sqrt(2) *
                                             1.001, include_zero=False)
        assert 6 + 12 == len(computer.get_vectors())

        computer = VectorCombinationComputer(self.input, 8)
        assert len(computer.get_vectors()) == len(computer.get_vectors())

    def test_collection_oblique(self):
        self.input[1][0] = 8

        computer = VectorCombinationComputer(self.input, 1)
        assert 7 == len(computer.get_vectors())

        computer = VectorCombinationComputer(self.input, math.sqrt(2) * 1.001)
        assert 1 + 6 + 12 == len(computer.get_vectors())