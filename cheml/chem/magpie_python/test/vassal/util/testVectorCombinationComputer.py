import unittest
import numpy as np
import numpy.testing as np_tst
from vassal.util.VectorCombinationComputer import VectorCombinationComputer
import math

class testVectorCombinationComputer(unittest.TestCase):
    def setUp(self):
        self.input = list(np.identity(3))

    def tearDown(self):
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
        self.assertEquals(7, len(computer.get_vectors()))

        computer = VectorCombinationComputer(self.input, math.sqrt(2) * 1.001)
        self.assertEquals(1 + 6 + 12, len(computer.get_vectors()))

        computer = VectorCombinationComputer(self.input, math.sqrt(2) *
                                             1.001, include_zero=False)
        self.assertEquals(6 + 12, len(computer.get_vectors()))

        computer = VectorCombinationComputer(self.input, 8)
        self.assertEquals(len(computer.get_vectors()),
                          len(computer.get_vectors()))

    def test_collection_oblique(self):
        self.input[1][0] = 8

        computer = VectorCombinationComputer(self.input, 1)
        self.assertEquals(7, len(computer.get_vectors()))

        computer = VectorCombinationComputer(self.input, math.sqrt(2) * 1.001)
        self.assertEquals(1 + 6 + 12, len(computer.get_vectors()))