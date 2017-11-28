import unittest
import numpy as np
import numpy.testing as np_tst
from attributes.generators.composition.ElementPairPropertyAttributeGenerator \
    import ElementPairPropertyAttributeGenerator
from data.materials.CompositionEntry import CompositionEntry

class testElementPairPropertyAttributeGenerator(unittest.TestCase):
    def test_attribute_generator(self):
        # Make list of CompositionEntry's.
        entries = [CompositionEntry(composition="Fe"), CompositionEntry(
            composition="FeAl"), CompositionEntry(
            composition="Fe2AlZr"), CompositionEntry(
            composition="FeAl2Zr")]

        # Make generator and add property.
        el = ElementPairPropertyAttributeGenerator()
        el.add_elemental_pair_property("B2Volume")

        # Run the generator.
        features = el.generate_features(entries)

        # Test results.
        self.assertEquals(5, len(features.columns))

        # Fe.
        np_tst.assert_array_equal([np.nan] * 5, features.values[0])

        # AlFe
        np_tst.assert_array_almost_equal([11.8028, 11.8028, 0, 11.8028, 0],
                                         features.values[1])

        # Fe2AlZr
        np_tst.assert_array_almost_equal([19.3989, 11.8028, 7.5961, 14.94118,
                                    2.510704], features.values[2])

        # FeAl2Zr
        np_tst.assert_array_almost_equal([19.3989, 11.8028, 7.5961, 15.65082,
                                          3.078416], features.values[3])