import os
from importlib.resources import files

from chemml.chem.magpie_python.data.materials.CompositionEntry import CompositionEntry
from chemml.chem.magpie_python.data.materials.util.GCLPCalculator import GCLPCalculator
from chemml.chem.magpie_python.data.materials.util.LookUpData import LookUpData

class TestGCLPCalculator:
    # this_file_path = os.path.dirname(__file__)
    # abs_path = os.path.join(this_file_path, "../../../test-files/")
    abs_path = str(files('chemml').joinpath('datasets', 'data', 'magpie_python_test'))
    def setup_method(self):
        self.calc = GCLPCalculator()

    def teardown_method(self):
        self.calc = None

    def test_initialization(self):
        n_elem = self.calc.get_num_phases()
        assert len(LookUpData.element_names) == n_elem, 'Initial number of phases should be equal to 112'
        # Add in NaCl.
        NaCl = CompositionEntry("NaCl")
        self.calc.add_phase(NaCl, -1)
        assert 1 + n_elem == self.calc.get_num_phases()

        # Add in a duplicate.
        self.calc.add_phase(NaCl, -1)
        assert 1 + n_elem == self.calc.get_num_phases()

        # See if energy is updated.
        self.calc.add_phase(NaCl, 0)
        assert abs(-1 - self.calc.phases[NaCl]) <= 1e-06
        self.calc.add_phase(NaCl, -2)
        assert abs(-2 - self.calc.phases[NaCl]) <= 1e-06

        # Add many phases.
        entries = CompositionEntry.import_composition_list(
            os.path.join(self.abs_path, "small_set_comp.txt"))
        energies = CompositionEntry.import_values_list(
            os.path.join(self.abs_path, "small_set_delta_e.txt"))
        self.calc.add_phases(entries, energies)

        assert 725 == self.calc.get_num_phases(), 'Total number of phases should be equal.'

    def test_GCLP(self):
        n_elem = self.calc.get_num_phases()
        assert len(LookUpData.element_names) == n_elem, 'Initial number of phases should be equal to 112'

        # Simple test: No phases.
        NaCl = CompositionEntry("NaCl")
        left, right = self.calc.run_GCLP(NaCl)
        assert abs(0.0 - left) <= 1e-06
        assert 2 == len(right)

        # Add in Na2Cl and NaCl2 to map.
        self.calc.add_phase(CompositionEntry("Na2Cl"), -1)
        self.calc.add_phase(CompositionEntry("NaCl2"), -1)
        left, right = self.calc.run_GCLP(NaCl)
        assert abs(-1 - left) <= 1e-06
        assert 2 == len(right)

        # Add NaCl to the map.
        self.calc.add_phase(NaCl, -2)
        left, right = self.calc.run_GCLP(NaCl)
        assert abs(-2 - left) <= 1e-06
        assert 1 == len(right)

        # Make sure it can do systems not included in original.
        left, right = self.calc.run_GCLP(CompositionEntry(
            "AlNiFeZrTiSiBrFOSeKHHe"))
        assert abs(0.0 - left) <= 1e-06
        assert 13 == len(right)