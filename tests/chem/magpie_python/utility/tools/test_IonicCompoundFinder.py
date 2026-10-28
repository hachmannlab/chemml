from chemml.chem.magpie_python.data.materials.CompositionEntry import CompositionEntry
from chemml.chem.magpie_python.utility.tools.IonicCompoundFinder import IonicCompoundFinder

class TestIonicCompoundFinder:
    def setup_method(self):
        self.icf = IonicCompoundFinder()

    def teardown_method(self):
        self.icf = None

    def test_NaCl(self):
        self.icf.set_nominal_composition(CompositionEntry(composition="NaCl"))
        self.icf.set_maximum_distance(0.2)
        self.icf.set_max_formula_unit_size(4)

        # Make sure it finds only one.
        accepted = self.icf.find_all_compounds()
        assert 1 == len(accepted)

    def test_FeO(self):
        self.icf.set_nominal_composition(CompositionEntry(composition="Fe2O"))
        self.icf.set_maximum_distance(0.34)
        self.icf.set_max_formula_unit_size(5)

        # Make sure it finds only one (FeO).
        accepted = self.icf.find_all_compounds()
        assert 1 == len(accepted)

        # Make sure it finds two (FeO, Fe2O3).
        self.icf.set_maximum_distance(0.54)
        accepted = self.icf.find_all_compounds()
        assert 2 == len(accepted)
        assert 'FeO' == accepted[0].__str__()

    def test_NaBrCl(self):
        self.icf.set_nominal_composition(CompositionEntry(
            composition="Na2.1ClBr"))
        self.icf.set_maximum_distance(0.1)
        self.icf.set_max_formula_unit_size(5)

        # Make sure it finds only one (Na2ClBr).
        accepted = self.icf.find_all_compounds()
        assert 1 == len(accepted)

    def test_Ba2As2S(self):
        self.icf.set_nominal_composition(CompositionEntry(
            composition="Ba2As2S"))
        self.icf.set_maximum_distance(0.35)
        self.icf.set_max_formula_unit_size(7)

        # Make sure it finds Ba4As2S.
        accepted = self.icf.find_all_compounds()
        assert CompositionEntry(composition='Ba4As2S') in accepted