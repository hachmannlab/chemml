import unittest
import os
import numpy as np
import numpy.testing as np_test
from attributes.generators.composition.ElementalPropertyAttributeGenerator \
    import ElementalPropertyAttributeGenerator
from data.materials.CompositionEntry import CompositionEntry

class testElementalPropertyAttributeGenerator(unittest.TestCase):
    this_file_path = os.path.dirname(__file__)
    rel_path = os.path.join(this_file_path, "../../../../lookup-data/")

    def test_easy(self):
        # Make list of CompositionEntry's.
        entries = [CompositionEntry(composition="NaCl"), CompositionEntry(
                    composition="Fe2O3")]

        # Make generator and add property.
        el = ElementalPropertyAttributeGenerator(use_default_properties=False)
        el.add_elemental_property("Number")

        # Run generator.
        features = el.generate_features(entries, lookup_path=self.rel_path)

        # Test results.
        self.assertEquals(6, features.values[0].size)

        # Results for NaCl.
        np_test.assert_array_almost_equal([14, 6, 3, 17, 11, 14],
                                          features.values[0])

        # Results for Fe2O3.
        np_test.assert_array_almost_equal([15.2, 18, 8.64, 26, 8, 8],
                                          features.values[1])

    def test_with_missing(self):
        # Make list of CompositionEntry's.
        entries = [CompositionEntry(composition="HBr")]

        # Make generator and add properties.
        el = ElementalPropertyAttributeGenerator(use_default_properties=False)
        el.add_elemental_property("Number")
        el.add_elemental_property("ZungerPP-r_d")
        el.add_elemental_property("Row")

        # Run generator.
        features = el.generate_features(entries, lookup_path=self.rel_path)

        # Test results.
        self.assertEquals(18, features.values[0].size)

        # Results for HBr, only bothering testing mean and max.
        self.assertAlmostEquals(18, features.values[0][0], delta=1e-6)
        self.assertAlmostEquals(35, features.values[0][3], delta=1e-6)
        self.assertTrue(np.isnan(features.values[0][6]))
        self.assertTrue(np.isnan(features.values[0][9]))
        self.assertAlmostEquals(2.5, features.values[0][12], delta=1e-6)
        self.assertAlmostEquals(4, features.values[0][15], delta=1e-6)