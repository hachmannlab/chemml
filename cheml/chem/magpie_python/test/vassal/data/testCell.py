import math
import unittest
from numpy.linalg import norm
import numpy as np
import numpy.testing as np_tst
from ....vassal.data.Atom import Atom
from ....vassal.data.Cell import Cell

class testCell(unittest.TestCase):
    cell = None

    # Create one instance per test.
    def setUp(self):
        self.cell = Cell()

    # Destroy instance as soon as test is over.
    def tearDown(self):
        self.cell = None

    def test_set_basis(self):
        # Test using angles and lattice parameters as input.
        self.cell.set_basis(lengths=[5.643, 6.621,4.885], angles=[91.83,
                            93.58, 107.69])
        self.assertAlmostEquals(173.30, self.cell.volume(), delta=1e-2)
        np_tst.assert_array_almost_equal([5.643, 6.621,4.885],
                                         self.cell.get_lattice_parameters())
        np_tst.assert_array_almost_equal([91.83, 93.58, 107.69],
                    self.cell.get_lattice_angles_radians(radians=False))

        # Simple test with a primitive cell.
        basis = np.zeros((3, 3))
        basis[0] = np.array([0, 2.986, 2.986])
        basis[1] = np.array([2.986, 0, 2.986])
        basis[2] = np.array([2.986, 2.986, 0])

        self.cell.set_basis(basis=basis)
        self.assertAlmostEquals(13.312*4, self.cell.volume(), delta=1e-3)
        np_tst.assert_array_almost_equal([4.223, 4.223, 4.223],
                                         self.cell.get_lattice_parameters(),
                                         decimal=3)
        np_tst.assert_array_almost_equal([60, 60, 60],
                    self.cell.get_lattice_angles_radians(radians=False))

    def test_aligned_basis(self):
        # Simple test with a primitive cell.
        basis = np.zeros((3, 3))
        basis[0] = np.array([0, 2.986, 2.986])
        basis[1] = np.array([2.986, 0, 2.986])
        basis[2] = np.array([2.986, 2.986, 0])

        self.cell.set_basis(basis=basis)

        # Compute the aligned basis.
        aligned_basis = self.cell.get_aligned_basis()
        self.assertAlmostEquals(0, aligned_basis[1][0], delta=1e-6)
        self.assertAlmostEquals(0, aligned_basis[2][0], delta=1e-6)
        self.assertAlmostEquals(0, aligned_basis[2][1], delta=1e-6)

    def test_clone(self):
        self.cell.add_atom(Atom([0, 0, 0], 0))
        self.cell.set_type_name(0, "A")

        # Test adding atoms.
        clone = self.cell.__copy__()
        self.assertEquals(clone, self.cell)
        clone.add_atom(Atom([0, 0.5, 0], 0))
        self.assertFalse(clone.__eq__(self.cell))

        # Test changing atom.
        clone = self.cell.__copy__()
        clone.get_atom(0).set_type(1)
        self.assertFalse(clone.__eq__(self.cell))

        # Test changing basis.
        clone = self.cell.__copy__()
        clone.set_basis(lengths=[2, 1, 1], angles=[90, 90, 90])
        self.assertFalse(clone.__eq__(self.cell))

    def test_lattice_vectors(self):
       self.cell.set_basis(lengths=[1, 2, 3], angles=[80, 90, 70])
       l_vec = self.cell.get_lattice_vectors()
       np_tst.assert_array_almost_equal([[1.0, 0.0, 0.0], [0.684, 1.879,
                0.0], [0.0, 0.554, 2.948]], l_vec, decimal=3)

       # FCC primitive cell.
       self.cell.set_basis(lengths=[0.70710678118655, 0.70710678118655,
                                    0.70710678118655], angles=[60, 60, 60])
       self.assertAlmostEquals(0.25, self.cell.volume(), delta=1e-6)
       l_vec = self.cell.get_lattice_vectors()
       self.assertAlmostEquals(0.70710678118655, norm(l_vec[0]),
                               delta=1e-2)

    def test_fractional_to_cartesian(self):
        self.cell.set_basis(lengths=[1, 2, 3], angles=[80, 90, 70])
        np_tst.assert_array_almost_equal([0.2368, 0.5421, 0.8844],
                self.cell.convert_fractional_to_cartesian([0.1, 0.2, 0.3]),
                                         decimal=3)

    def test_cartesian_to_fractional(self):
        self.cell.set_basis(lengths=[1, 2, 3], angles=[80, 90, 70])
        np_tst.assert_array_almost_equal([0.1, 0.2, 0.3],
            self.cell.convert_cartesian_to_fractional([0.2368, 0.5421, 0.8844]),
                                         decimal=3)

    def test_supercell_translation(self):
        self.cell.set_basis(lengths=[0.70710678118655, 0.70710678118655,
                                     0.70710678118655], angles=[60, 60, 60])
        self.assertAlmostEquals(0.25, self.cell.volume(), delta=1e-6)
        l_vec = self.cell.get_lattice_vectors()

        # Check a few.
        pos = self.cell.get_periodic_image([0, 0, 0], 1, 0, 0)
        np_tst.assert_array_almost_equal([0.70710678000000, 0, 0], pos,
                                         decimal=3)
        pos = self.cell.get_periodic_image([0, 0, 0], 1, 1, 0)
        np_tst.assert_array_almost_equal([1.06066017000000, 0.61237243466821,
                                0], pos, decimal=3)
        pos = self.cell.get_periodic_image([0, 0, 0], 1, 1, 1)
        np_tst.assert_array_almost_equal([1.41421356000000, 0.81649657955762,
                                0.57735026918963], pos, decimal=3)

    def test_equals(self):
        # Make other cell
        other = Cell()

        # First check.
        self.assertTrue(self.cell.__eq__(other))

        # Adjust basis.
        self.cell.set_basis(lengths=[1, 2, 3], angles=[70, 80, 90])
        self.assertFalse(self.cell.__eq__(other))
        other.set_basis(lengths=[1, 2, 3], angles=[70, 80, 90])
        self.assertTrue(self.cell.__eq__(other))

        # Add an atom to 0,0,0
        self.cell.add_atom(Atom([0, 0, 0], 0))
        self.assertFalse(self.cell.__eq__(other))
        other.add_atom(Atom([0, 0, 0], 0))
        self.assertTrue(self.cell.__eq__(other))

        # Changing names.
        self.cell.set_type_name(0, "Al")
        self.assertFalse(self.cell.__eq__(other))
        other.set_type_name(0, "Al")
        self.assertTrue(self.cell.__eq__(other))

        # Adding more atoms of different type.
        self.cell.add_atom(Atom([0.5, 0.5, 0], 1))
        other.add_atom(Atom([0.5, 0.5, 0], 0))
        self.assertFalse(self.cell.__eq__(other))
        other.get_atom(1).set_type(1)
        self.assertTrue(self.cell.__eq__(other))

        # Adding atoms with different positions.
        self.cell.add_atom(Atom([0.5, 0, 0.5], 1))
        other.add_atom(Atom([0, 0.5, 0.5], 1))
        self.assertFalse(self.cell.__eq__(other))

        # Adding atoms out of sequence.
        other.add_atom(Atom([0.5, 0, 0.5], 1))
        self.cell.add_atom(Atom([0, 0.5, 0.5], 1))
        self.assertTrue(self.cell.__eq__(other))

    def test_minimum_distance(self):
        # Simple case: orthogonal axes.

        # Origin.
        self.cell.add_atom(Atom([0, 0, 0], 1))
        # C face center.
        self.cell.add_atom(Atom([0.5, 0.5, 0], 1))

        dist = self.cell.get_minimum_distance(point1=[0, 0, 0], point2=[0.5,
                                                            0.5, 0])
        self.assertAlmostEquals(math.sqrt(0.5), dist, delta=1e-6)
        dist = self.cell.get_minimum_distance(point1=[0, 0, 0], point2=[2.5,
                                                            0.5, -10])
        self.assertAlmostEquals(math.sqrt(0.5), dist, delta=1e-6)

        # Difficult case: Non-conventional unit cell.
        basis = self.cell.get_basis()
        basis[1][0] = 108
        self.cell.set_basis(basis=basis)
        dist = self.cell.get_minimum_distance(point1=[0, 0, 0], point2=[0.5,
                                                            0.5, 0])
        self.assertAlmostEquals(math.sqrt(0.5), dist, delta=1e-6)
        dist = self.cell.get_minimum_distance(point1=[0, 0, 0], point2=[5.5,
                                                            0.5, 0])
        self.assertAlmostEquals(math.sqrt(0.5), dist, delta=1e-6)
        dist = self.cell.get_minimum_distance(point1=[0, 0, 0], point2=[5.5,
                                                            -10.5, 0])
        self.assertAlmostEquals(math.sqrt(0.5), dist, delta=1e-6)

    def test_get_closest_image_simple(self):
        # Simple case: orthogonal axes.

        # Origin.
        self.cell.add_atom(Atom([0, 0, 0], 1))
        # C face center.
        self.cell.add_atom(Atom([0.75, 0.75, 0.75], 1))
        image = self.cell.get_minimum_distance(center=0, neighbor=1)
        np_tst.assert_array_almost_equal([-0.25, -0.25, -0.25],
                                         image.get_position(), decimal=6)
        np_tst.assert_array_equal([-1, -1, -1], image.get_supercell())

    def test_get_closest_image_difficult(self):
        # Difficult case: Non-conventional unit cell.
        # Origin.
        self.cell.add_atom(Atom([0, 0, 0], 1))
        # Body face center.
        self.cell.add_atom(Atom([0.5, 0.5, 0.5], 1))
        basis = self.cell.get_basis()
        basis[1][0] = 108
        self.cell.set_basis(basis=basis)
        image = self.cell.get_minimum_distance(center=0, neighbor=1)
        np_tst.assert_array_almost_equal([-0.5, -0.5, 0.5],
                                         image.get_position(), decimal=6)
        np_tst.assert_array_equal([-1, 53, 0], image.get_supercell())

    def test_replacement(self):
        # Make the original cell B2-CuZr
        self.cell.add_atom(Atom([0, 0, 0], 0))
        self.cell.add_atom(Atom([0.5, 0.5, 0.5], 1))
        self.cell.set_type_name(0, "Cu")
        self.cell.set_type_name(1, "Zr")

        # Replace Cu with Ni.
        to_change = {"Cu":"Ni"}
        self.cell.replace_type_names(to_change)
        self.assertEquals("Ni", self.cell.get_type_name(0))
        self.assertEquals("Zr", self.cell.get_type_name(1))

        # Replace Ni with Cu and Zr with Ti.
        to_change = {"Ni": "Cu", "Zr":"Ti"}
        self.cell.replace_type_names(to_change)
        self.assertEquals("Cu", self.cell.get_type_name(0))
        self.assertEquals("Ti", self.cell.get_type_name(1))

        # Exchange Cu and Ti.
        to_change = {"Ti": "Cu", "Cu": "Ti"}
        self.cell.replace_type_names(to_change)
        self.assertEquals("Ti", self.cell.get_type_name(0))
        self.assertEquals("Cu", self.cell.get_type_name(1))

        # Make everything Cu.
        to_change = {"Ti": "Cu"}
        self.cell.replace_type_names(to_change)
        self.assertEquals("Cu", self.cell.get_type_name(0))
        self.assertEquals("Cu", self.cell.get_type_name(1))

        # Make everything W.
        to_change = {"Cu":"W"}
        self.cell.replace_type_names(to_change)
        self.assertEquals("W", self.cell.get_type_name(0))
        self.assertEquals("W", self.cell.get_type_name(1))

        # Merge types.
        self.cell.merge_like_types()
        self.assertEquals(1, self.cell.n_types())