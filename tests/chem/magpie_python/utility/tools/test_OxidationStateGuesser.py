import unittest
import numpy.testing as np
from chemml.chem.magpie_python.data.materials.CompositionEntry import CompositionEntry
from chemml.chem.magpie_python.data.materials.util.LookUpData import LookUpData
from chemml.chem.magpie_python.utility.tools.OxidationStateGuesser import OxidationStateGuesser

class testOxidationStateGuesser(unittest.TestCase):
    def test_guesser(self):
        ox = OxidationStateGuesser()
        ox.set_oxidationstates(LookUpData.load_property("OxidationStates"))
        ox.set_electronegativity(LookUpData.load_property("Electronegativity"))

        res = ox.get_possible_states(CompositionEntry(composition="NaCl"))
        self.assertEqual(1, len(res))
        np.assert_array_equal([1, -1], res[0])

        res = ox.get_possible_states(CompositionEntry(composition="Fe2O3"))
        self.assertEqual(1, len(res))
        np.assert_array_equal([3, -2], res[0])

        res = ox.get_possible_states(CompositionEntry(composition="NaHCO3"))
        self.assertEqual(1, len(res))
        np.assert_array_equal([10, 0, 5, 7], CompositionEntry(
            composition="NaHCO3").get_element_ids())
        np.assert_array_equal([1, 1, 4, -2], res[0])

        res = ox.get_possible_states(CompositionEntry(composition="NH3"))
        self.assertEqual(2, len(res))
        np.assert_array_equal([0, 6], CompositionEntry(
            composition="NH3").get_element_ids())
        np.assert_array_equal([1, -3], res[0])

        res = ox.get_possible_states(CompositionEntry(composition="NaAl"))
        self.assertEqual(0, len(res))

        res = ox.get_possible_states(CompositionEntry(composition="PbTiO3"))
        self.assertEqual(2, len(res))
        np.assert_array_equal([21, 81, 7], CompositionEntry(
            composition="PbTiO3").get_element_ids())
        np.assert_array_equal([4, 2, -2], res[0])