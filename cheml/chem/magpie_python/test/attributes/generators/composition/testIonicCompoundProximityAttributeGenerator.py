import unittest
import numpy.testing as np_tst
from attributes.generators.composition\
    .IonicCompoundProximityAttributeGenerator import \
    IonicCompoundProximityAttributeGenerator
from data.materials.CompositionEntry import CompositionEntry

class testIonicCompoundProximityAttributeGenerator(unittest.TestCase):
    def test_attribute_generator(self):
        # Make a list of CompositionEntry's.
        entries = [CompositionEntry(composition="Na0.9Cl1.1"),
                   CompositionEntry(composition="CuCl2"), CompositionEntry(
                composition="FeAl"), CompositionEntry(composition="Fe")]

        # Make the generator and set options.
        ig = IonicCompoundProximityAttributeGenerator()
        ig.set_max_formula_unit(10)

        # Run the generator.
        features = ig.generate_features(entries)

        # Test results.
        self.assertEquals(1, len(features.columns))
        np_tst.assert_array_almost_equal([0.1, 0, 2, 1], features.ix[:,0])

        # Now decrease the maximum size to 2, which means CuCl2 should match
        # CuCl.
        ig.set_max_formula_unit(2)

        # Run the generator.
        features = ig.generate_features(entries)

        # Test results.
        self.assertEquals(1, len(features.columns))
        np_tst.assert_array_almost_equal([0.1, 1.0 / 3, 2, 1],
                                         features.ix[:,0])