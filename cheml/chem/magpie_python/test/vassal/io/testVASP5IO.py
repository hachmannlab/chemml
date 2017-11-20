import unittest
import os
import numpy.testing as np_tst
from vassal.data.Atom import Atom
from vassal.data.Cell import Cell
from vassal.io.VASP5IO import VASP5IO

class testVASP5IO(unittest.TestCase):
    def test_conversion(self):
        # Make an FCC cell.
        cell = Cell()
        cell.set_basis(lengths=[3.5, 3.6, 3.4], angles=[89, 90, 91])
        cell.add_atom(Atom([0, 0, 0], 0))
        cell.add_atom(Atom([0.5, 0.5, 0], 1))
        cell.add_atom(Atom([0.5, 0, 0.5], 1))
        cell.add_atom(Atom([0, 0.5, 0.5], 1))
        cell.set_type_name(0, "Al")
        cell.set_type_name(1, "Ni")

        # Convert it to string.
        vio = VASP5IO()
        temp = vio.convert_structure_to_string(cell)

        # Convert it back.
        new_cell = vio.parse_file(list_of_lines=temp)

        # Check to make sure everything is good.
        self.assertAlmostEquals(cell.volume(), new_cell.volume(), delta=1e-4)
        self.assertEquals(cell.n_types(), new_cell.n_types())
        np_tst.assert_array_almost_equal(cell.get_lattice_vectors()[1],
                                         new_cell.get_lattice_vectors()[1],
                                         decimal=4)
        new_temp = vio.convert_structure_to_string(new_cell)
        np_tst.assert_equal(temp, new_temp)

    def test_parse_from_file(self):
        vio = VASP5IO()
        this_file_path = os.path.dirname(__file__)
        rel_path = os.path.join(this_file_path, "../../test-files/")
        cell = vio.parse_file(file_name=rel_path+"393-Ta1.vasp")
        self.assertAlmostEquals(556.549, cell.volume(), delta=1e-2)
        self.assertAlmostEquals(10.218, cell.get_lattice_vectors()[0][0],
                                delta=1e-2)
        self.assertEquals(30, cell.n_atoms())
        np_tst.assert_array_almost_equal([0.681, 0.818, 0.998],
                                         cell.get_atom(29).get_position(),
                                         decimal=2)