import unittest
import os
from .....data.materials.CompositionEntry import CompositionEntry
from .....data.materials.util.GCLPCalculator import GCLPCalculator
from .....data.materials.util.LookUpData import LookUpData

class testGCLPCalculator(unittest.TestCase):
    this_file_path = os.path.dirname(__file__)
    abs_path = os.path.join(this_file_path, "../../../test-files/")
    def setUp(self):
        self.calc = GCLPCalculator()

    def tearDown(self):
        self.calc = None

    def test_initialization(self):
        n_elem = self.calc.get_num_phases()
        self.assertEquals(len(LookUpData.element_names), n_elem, "Initial "
                                                                 "number"
                                        " of phases should be equal to 112")
        # Add in NaCl.
        NaCl = CompositionEntry("NaCl")
        self.calc.add_phase(NaCl, -1)
        self.assertEquals(1 + n_elem, self.calc.get_num_phases())

        # Add in a duplicate.
        self.calc.add_phase(NaCl, -1)
        self.assertEquals(1 + n_elem, self.calc.get_num_phases())

        # See if energy is updated.
        self.calc.add_phase(NaCl, 0)
        self.assertAlmostEquals(-1, self.calc.phases[NaCl], delta=1e-6)
        self.calc.add_phase(NaCl, -2)
        self.assertAlmostEquals(-2, self.calc.phases[NaCl], delta=1e-6)

        # Add many phases.
        entries = CompositionEntry.import_composition_list(
            self.abs_path+"small_set_comp.txt")
        energies = CompositionEntry.import_values_list(
            self.abs_path+"small_set_delta_e.txt")
        self.calc.add_phases(entries, energies)

        self.assertEquals(725, self.calc.get_num_phases(),
                          "Total number of phases should be equal.")

    def test_GCLP(self):
        n_elem = self.calc.get_num_phases()
        self.assertEqual(len(LookUpData.element_names), n_elem, "Initial number"
                                        " of phases should be equal to 112")

        # Simple test: No phases.
        NaCl = CompositionEntry("NaCl")
        left, right = self.calc.run_GCLP(NaCl)
        self.assertAlmostEquals(0.0, left, delta=1e-6)
        self.assertEquals(2, len(right))

        # Add in Na2Cl and NaCl2 to map.
        self.calc.add_phase(CompositionEntry("Na2Cl"), -1)
        self.calc.add_phase(CompositionEntry("NaCl2"), -1)
        left, right = self.calc.run_GCLP(NaCl)
        self.assertAlmostEquals(-1, left, delta=1e-6)
        self.assertEquals(2, len(right))

        # Add NaCl to the map.
        self.calc.add_phase(NaCl, -2)
        left, right = self.calc.run_GCLP(NaCl)
        self.assertAlmostEquals(-2, left, delta=1e-6)
        self.assertEquals(1, len(right))

        # Make sure it can do systems not included in original.
        left, right = self.calc.run_GCLP(CompositionEntry(
            "AlNiFeZrTiSiBrFOSeKHHe"))
        self.assertAlmostEquals(0.0, left, delta=1e-6)
        self.assertEquals(13, len(right))