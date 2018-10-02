# -*- coding: utf-8 -*-
import unittest
import numpy as np
import numpy.testing as np_test
from chemml.chem.magpie_python import ElementalPropertyAttributeGenerator
from chemml.chem.magpie_python import CompositionEntry

class testElementalPropertyAttributeGenerator(unittest.TestCase):

    def test_easy(self):
        # Make list of CompositionEntry's.
        entries = [CompositionEntry(composition="NaCl"), CompositionEntry(
                    composition="Fe2O3")]

        # Make generator and add property.
        el = ElementalPropertyAttributeGenerator(use_default_properties=False)
        el.add_elemental_property("Number")

        # Run generator.
        features = el.generate_features(entries)

        # Test results.
        self.assertEqual(6, features.values[0].size)

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
        features = el.generate_features(entries)

        # Test results.
        self.assertEqual(18, features.values[0].size)

        # Results for HBr, only bothering testing mean and max.
        self.assertAlmostEqual(18, features.values[0][0], delta=1e-6)
        self.assertAlmostEqual(35, features.values[0][3], delta=1e-6)
        self.assertTrue(np.isnan(features.values[0][6]))
        self.assertTrue(np.isnan(features.values[0][9]))
        self.assertAlmostEqual(2.5, features.values[0][12], delta=1e-6)
        self.assertAlmostEqual(4, features.values[0][15], delta=1e-6)