import unittest
import os
from data.materials.CompositionEntry import CompositionEntry
from utility.tools.IonicCompoundFinder import IonicCompoundFinder

class testIonicCompoundFinder(unittest.TestCase):
    def setUp(self):
        self.icf = IonicCompoundFinder()
        this_file_path = os.path.dirname(__file__)
        self.rel_path = os.path.join(this_file_path, "../../../lookup-data/")

    def tearDown(self):
        self.icf = None

    def test_NaCl(self):
        self.icf.set_nominal_composition(CompositionEntry(composition="NaCl"))
        self.icf.set_maximum_distance(0.2)
        self.icf.set_max_formula_unit_size(4)

        # Make sure it finds only one.
        accepted = self.icf.find_all_compounds(lookup_path=self.rel_path)
        self.assertEquals(1, len(accepted))

    def test_FeO(self):
        self.icf.set_nominal_composition(CompositionEntry(composition="Fe2O"))
        self.icf.set_maximum_distance(0.34)
        self.icf.set_max_formula_unit_size(5)

        # Make sure it finds only one (FeO).
        accepted = self.icf.find_all_compounds(lookup_path=self.rel_path)
        self.assertEquals(1, len(accepted))

        # Make sure it finds two (FeO, Fe2O3).
        self.icf.set_maximum_distance(0.54)
        accepted = self.icf.find_all_compounds(lookup_path=self.rel_path)
        self.assertEquals(2, len(accepted))
        self.assertEquals("FeO", accepted[0].__str__())

    def test_NaBrCl(self):
        self.icf.set_nominal_composition(CompositionEntry(
            composition="Na2.1ClBr"))
        self.icf.set_maximum_distance(0.1)
        self.icf.set_max_formula_unit_size(5)

        # Make sure it finds only one (Na2ClBr).
        accepted = self.icf.find_all_compounds(lookup_path=self.rel_path)
        self.assertEquals(1, len(accepted))

    def test_Ba2As2S(self):
        self.icf.set_nominal_composition(CompositionEntry(
            composition="Ba2As2S"))
        self.icf.set_maximum_distance(0.35)
        self.icf.set_max_formula_unit_size(7)

        # Make sure it finds Ba4As2S.
        accepted = self.icf.find_all_compounds(lookup_path=self.rel_path)
        self.assertTrue(CompositionEntry(composition="Ba4As2S") in accepted)