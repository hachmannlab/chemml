import unittest
import numpy as np
import numpy.testing as np_tst
from attributes.generators.composition.ChargeDependentAttributeGenerator import \
    ChargeDependentAttributeGenerator
from data.materials.CompositionEntry import CompositionEntry

class testChargeDependentAttributeGenerator(unittest.TestCase):
    def test_attribute_generator(self):
        entries = [CompositionEntry(composition="NaCl"), CompositionEntry(
            composition="Fe"), CompositionEntry(composition="ZrO2"),
                   CompositionEntry(composition="UF6"), CompositionEntry(
                composition="Na2CoOSe")]
        cg = ChargeDependentAttributeGenerator()
        features = cg.generate_features(entries)

        # NaCl.
        np_tst.assert_array_almost_equal([-1, 1, 2, 1.0, 0, 5.139076, 349,
                                          3.16 - 0.93], features.values[0])

        # Fe.
        np_tst.assert_array_equal([np.nan] * features.values[1].size,
                                  features.values[1])

        # ZrO2.
        np_tst.assert_array_almost_equal([-2, 4, 6, 8.0 / 3, 8.0 / 9, 77.0639,
                                141 * 2, 3.44 - 1.33], features.values[2])

        # UF6.
        np_tst.assert_array_almost_equal([-1, 6, 7, 12.0 / 7, 1.224489796],
                                         features.values[3][:5])
        np_tst.assert_array_equal([np.nan] * 3, features.values[3][5:])

        # Na2CoOSe.
        np_tst.assert_array_almost_equal([-2, 2, 4, 1.6, 0.48, 5.139076 * 2 /
        3 + 24.96501 / 3, 141.0 + 195.0, 2.995 - 1.246666667],
                                         features.values[4])