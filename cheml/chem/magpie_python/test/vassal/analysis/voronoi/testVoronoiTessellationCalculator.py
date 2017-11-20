import unittest
import os
import numpy as np
import time
from vassal.analysis.voronoi.VoronoiTessellationCalculator import \
    VoronoiTessellationCalculator
from vassal.data.Atom import Atom
from vassal.data.Cell import Cell
import numpy.testing as np_tst
from vassal.io.VASP5IO import VASP5IO

class testVoronoiTessellationCalculator(unittest.TestCase):
    this_file_path = os.path.dirname(__file__)
    rel_path = os.path.join(this_file_path, "../../../test-files/")

    def test_simple_cubic(self):
        # Create the simulation cell.
        structure = Cell()
        atom = Atom([0, 0, 0], 0)
        structure.add_atom(atom)

        # Run tessellation.
        result = VoronoiTessellationCalculator.compute(structure,
                                                       radical=False)

        # Test results.
        self.assertEquals(structure.n_atoms(), len(result))
        self.assertEquals(6, len(result[0].get_faces()))
        self.assertAlmostEquals(structure.volume(), result[0].get_volume(),
                                delta=1e-6)
        poly_index = result[0].get_polyhedron_shape()
        self.assertEquals(6, poly_index[4])
        poly_index = result[0].get_coordination_shell_shape(result)
        self.assertEquals(6, poly_index[0])

        # Test out the nearest neighbor shells.
        # 1st NN shell.
        nns = result[0].get_neighbor_shell(result, 1)
        self.assertEquals(6, len(nns))

        # 2nd NN shell.
        nns = result[0].get_neighbor_shell(result, 2)
        self.assertEquals(18, len(nns))

        # 3rd - 5th NN shell.
        for s in range(3, 6):
            nns = result[0].get_neighbor_shell(result, s)
            for image in nns:
                cell = image.get_supercell()
                self.assertAlmostEquals(s, sum([abs(cell[i]) for i in range(
                    3)]), delta=1e-6)

        # Test path NNs.
        # 0th NN.
        paths = result[0].get_neighbors_by_walks(result, 0)
        self.assertEquals(1, len(paths))
        total_weight = sum(paths.values())
        self.assertAlmostEquals(1.0, total_weight, delta=1e-6)

        # 1st NN.
        paths = result[0].get_neighbors_by_walks(result, 1)
        self.assertEquals(6, len(paths))
        total_weight = sum(paths.values())
        np_tst.assert_array_almost_equal([1/6.0]*6, paths.values())
        self.assertAlmostEquals(1.0, total_weight, delta=1e-6)

        # 2nd NN.
        paths = result[0].get_neighbors_by_walks(result, 2)
        self.assertEquals(18, len(paths))
        total_weight = sum(paths.values())
        for k,v in paths.iteritems():
            if 2 in k.get_supercell() or -2 in k.get_supercell():
                self.assertAlmostEquals(1 / 30.0, v, delta=1e-6)
            else:
                self.assertAlmostEquals(2 / 30.0, v, delta=1e-6)
        self.assertAlmostEquals(1.0, total_weight, delta=1e-6)

    def test_BCC(self):
        # Create the simulation cell.
        structure = Cell()
        atom = Atom([0, 0, 0], 0)
        structure.add_atom(atom)
        atom = Atom([0.5, 0.5, 0.5], 0)
        structure.add_atom(atom)

        # Run tessellation.
        result = VoronoiTessellationCalculator.compute(structure,
                                                       radical=False)

        # Test results.
        self.assertEquals(structure.n_atoms(), len(result))
        for cell in result:
            self.assertTrue(cell.geometry_is_valid())
            self.assertEquals(14, len(cell.get_faces()))
            self.assertAlmostEquals(0.5, cell.get_volume(), delta=1e-6)
            poly_index = cell.get_polyhedron_shape()
            self.assertEquals(8, poly_index[6])
            poly_index = result[0].get_coordination_shell_shape(result)
            self.assertEquals(6, poly_index[4])
            self.assertEquals(8, poly_index[6])

    def test_FCC(self):
        # Create the simulation cell.
        structure = Cell()
        structure.add_atom(Atom([0, 0, 0], 0))
        structure.add_atom(Atom([0.5, 0.5, 0], 0))
        structure.add_atom(Atom([0.5, 0, 0.5], 0))
        structure.add_atom(Atom([0, 0.5, 0.5], 0))

        # Run tessellation.
        result = VoronoiTessellationCalculator.compute(structure,
                                                       radical=False)

        # Test results.
        self.assertEquals(structure.n_atoms(), len(result))
        for cell in result:
            self.assertTrue(cell.geometry_is_valid())
            self.assertEquals(12, len(cell.get_faces()))
            self.assertAlmostEquals(0.25, cell.get_volume(), delta=1e-6)
            poly_index = cell.get_polyhedron_shape()
            self.assertEquals(12, poly_index[4])
            poly_index = result[0].get_coordination_shell_shape(result)
            self.assertEquals(12, poly_index[4])

    def test_FCC_primitive(self):
        # Create the simulation cell.
        structure = Cell()
        structure.set_basis(lengths=[0.70710678118655, 0.70710678118655,
                                     1.0], angles=[45, 90, 60])
        structure.add_atom(Atom([0, 0, 0], 0))

        # Run tessellation.
        result = VoronoiTessellationCalculator.compute(structure,
                                                       radical=False)

        # Test results.
        self.assertEquals(structure.n_atoms(), len(result))
        self.assertTrue(result[0].geometry_is_valid())
        self.assertEquals(12, len(result[0].get_faces()))
        poly_index = result[0].get_polyhedron_shape()
        self.assertEquals(12, poly_index[4])
        poly_index = result[0].get_coordination_shell_shape(result)
        self.assertEquals(12, poly_index[4])

    def test_Ta(self):
        structure = VASP5IO.parse_file(file_name=self.rel_path+"393-Ta1.vasp")

        # Run tessellation.
        result = VoronoiTessellationCalculator.compute(structure,
                                                       radical=False)

        # Test results.
        total_vol = 0.0
        for cell in result:
            total_vol += cell.get_volume()
            self.assertTrue(cell.geometry_is_valid())
        self.assertAlmostEquals(structure.volume(), total_vol,
                                delta=total_vol * 0.05)

    def test_Zr(self):
        structure = VASP5IO.parse_file(file_name=self.rel_path+"1214-Zr1.vasp")

        # Run tessellation.
        result = VoronoiTessellationCalculator.compute(structure,
                                                       radical=False)

        # Test results.
        total_vol = 0.0
        for cell in result:
            total_vol += cell.get_volume()
            self.assertTrue(cell.geometry_is_valid())
        self.assertAlmostEquals(structure.volume(), total_vol,
                                    delta=total_vol * 0.01)

    def test_C(self):
        structure = VASP5IO.parse_file(file_name=self.rel_path+"1004-C1.vasp")

        # Run tessellation.
        result = VoronoiTessellationCalculator.compute(structure,
                                                       radical=False)

        # Test results.
        total_vol = 0.0
        for cell in result:
            total_vol += cell.get_volume()
            self.assertTrue(cell.geometry_is_valid())
        self.assertAlmostEquals(structure.volume(), total_vol,
                                    delta=total_vol * 0.01)

    def test_C2(self):
        structure = VASP5IO.parse_file(file_name=self.rel_path+"846-C1.vasp")

        # Run tessellation.
        result = VoronoiTessellationCalculator.compute(structure,
                                                       radical=False)

        # Test results.
        total_vol = 0.0
        for cell in result:
            total_vol += cell.get_volume()
            self.assertTrue(cell.geometry_is_valid())
        vol_error = (total_vol - structure.volume()) / structure.volume()
        self.assertAlmostEquals(0.0, vol_error, delta=1e-2)

    def test_Ho(self):
        structure = VASP5IO.parse_file(file_name=self.rel_path+"592-Ho1.vasp")

        # Run tessellation.
        result = VoronoiTessellationCalculator.compute(structure,
                                                       radical=False)

        # Test results.
        total_vol = 0.0
        for cell in result:
            total_vol += cell.get_volume()
            self.assertTrue(cell.geometry_is_valid())
        vol_error = (total_vol - structure.volume()) / structure.volume()
        self.assertAlmostEquals(0.0, vol_error, delta=1e-2)

    def test_Si(self):
        structure = VASP5IO.parse_file(file_name=self.rel_path+"399-Si1.vasp")

        # Run tessellation.
        result = VoronoiTessellationCalculator.compute(structure,
                                                       radical=False)

        # Test results.
        total_vol = 0.0
        for cell in result:
            total_vol += cell.get_volume()
            self.assertTrue(cell.geometry_is_valid())
        vol_error = (total_vol - structure.volume()) / structure.volume()
        self.assertAlmostEquals(0.0, vol_error, delta=1e-2)

    def test_Li(self):
        structure = VASP5IO.parse_file(file_name=self.rel_path+"478-Li1.vasp")

        # Run tessellation.
        result = VoronoiTessellationCalculator.compute(structure,
                                                       radical=False)

        # Test results.
        total_vol = 0.0
        for cell in result:
            total_vol += cell.get_volume()
            self.assertTrue(cell.geometry_is_valid())
        vol_error = (total_vol - structure.volume()) / structure.volume()
        self.assertAlmostEquals(0.0, vol_error, delta=1e-2)

    def test_B(self):
        structure = VASP5IO.parse_file(file_name=self.rel_path+"673-B1.vasp")

        # Run tessellation.
        result = VoronoiTessellationCalculator.compute(structure,
                                                       radical=False)

        # Test results.
        total_vol = 0.0
        for cell in result:
            total_vol += cell.get_volume()
            self.assertTrue(cell.geometry_is_valid())
        self.assertAlmostEquals(structure.volume(), total_vol,
                                    delta=total_vol * 0.01)

    def test_ICSD_examples(self):
        example = ["3315-Ge2Os2Th1", "1001-N1Y1", "11375-C2N1",
            "12012-Ge2Ru2Tb1", "3778-Sr1Zn2", "4746-Cd1Cu4Er1"]
        for e in example:
            structure = VASP5IO.parse_file(file_name=self.rel_path+e+".vasp")

            # Run tessellation.
            result = VoronoiTessellationCalculator.compute(structure,
                                                           radical=False)

            # Test results.
            total_vol = 0.0
            for cell in result:
                total_vol += cell.get_volume()
                self.assertTrue(cell.geometry_is_valid())
            vol_error = (total_vol - structure.volume()) / structure.volume()
            self.assertAlmostEquals(0.0, vol_error, delta=1e-2)

    def test_Hg2K(self):
        structure = VASP5IO.parse_file(
            file_name=self.rel_path+"7823-Hg2K1.vasp")

        # Run tessellation.
        result = VoronoiTessellationCalculator.compute(structure,
                                                       radical=False)

        # Test results.
        total_vol = 0.0
        for cell in result:
            total_vol += cell.get_volume()
            self.assertTrue(cell.geometry_is_valid())
        vol_error = (total_vol - structure.volume()) / structure.volume()
        self.assertAlmostEquals(0.0, vol_error, delta=1e-2)

    def test_Ag2Pr1Si2(self):
        structure = VASP5IO.parse_file(
            file_name=self.rel_path+"8379-Ag2Pr1Si2.vasp")

        # Run tessellation.
        result = VoronoiTessellationCalculator.compute(structure,
                                                       radical=False)

        # Test results.
        total_vol = 0.0
        for cell in result:
            total_vol += cell.get_volume()
            self.assertTrue(cell.geometry_is_valid())
        vol_error = (total_vol - structure.volume()) / structure.volume()
        self.assertAlmostEquals(0.0, vol_error, delta=1e-2)

    def test_Sc(self):
        structure = VASP5IO.parse_file(file_name=self.rel_path+"1565-Sc1.vasp")

        # Run tessellation.
        result = VoronoiTessellationCalculator.compute(structure,
                                                       radical=False)

        # Test results.
        total_vol = 0.0
        for cell in result:
            total_vol += cell.get_volume()
            self.assertTrue(cell.geometry_is_valid())
        vol_error = (total_vol - structure.volume()) / structure.volume()
        self.assertAlmostEquals(0.0, vol_error, delta=1e-2)

    def test_big(self):
        # Number of atoms in each direction.
        n_atom = 10
        structure = Cell()
        structure.set_basis(lengths=[2 * n_atom, 2 * n_atom, 2 * n_atom],
                            angles=[90, 90, 90])

        # Add a bunch of atoms.
        step_size = 1.0 / n_atom
        for x in range(n_atom):
            for y in range(n_atom):
                for z in range(n_atom):
                    new_pos = np.array([x, y, z], dtype=float) + \
                              np.random.random(3) / n_atom
                    new_pos *= step_size
                    structure.add_atom(Atom(new_pos, 0))

        # Compute the cells.
        cells = VoronoiTessellationCalculator.compute(structure, radical=True)
        total_vol = 0.0
        for cell in cells:
            total_vol += cell.get_volume()
            self.assertTrue(cell.geometry_is_valid())

        vol_error = (total_vol - structure.volume()) / structure.volume()
        self.assertAlmostEquals(0.0, vol_error, delta=1e-2)

    def test_random_packing(self):
        # Number of atoms in each direction.
        n_atom = 10
        structure = Cell()
        structure.set_basis(lengths=[2 * n_atom, 2 * n_atom, 2 * n_atom],
                            angles=[90, 90, 90])

        # Add a bunch of atoms.
        for x in range(n_atom):
            for y in range(n_atom):
                for z in range(n_atom):
                    structure.add_atom(Atom(np.random.random(3), 0))

        # Compute the cells.
        cells = VoronoiTessellationCalculator.compute(structure, radical=True)
        total_vol = 0.0
        for cell in cells:
            total_vol += cell.get_volume()
            self.assertTrue(cell.geometry_is_valid())

        vol_error = (total_vol - structure.volume()) / structure.volume()
        self.assertAlmostEquals(0.0, vol_error, delta=1e-2)

    def test_scaling(self):
        print "size\ttime\tvol_error"
        n_atom_list = range(1, 9) + [10, 20]
        # Number of atoms in each direction.
        for n_atom in n_atom_list:
            structure = Cell()
            structure.set_basis(lengths=[2 * n_atom, 2 * n_atom, 2 * n_atom],
                                angles=[90, 90, 90])
            # Add a bunch of atoms.
            step_size = 1.0 / n_atom
            for x in range(n_atom):
                for y in range(n_atom):
                    for z in range(n_atom):
                        new_pos = np.array([x, y, z], dtype=float) + \
                                  np.random.random(3) / n_atom
                        new_pos *= step_size
                        structure.add_atom(Atom(new_pos, 0))

            # Compute the cells.
            start = time.time()
            cells = VoronoiTessellationCalculator.compute(structure,
                                                          radical=True)
            run_time = time.time() - start
            total_vol = 0.0
            for cell in cells:
                total_vol += cell.get_volume()
                self.assertTrue(cell.geometry_is_valid())

            vol_error = (total_vol - structure.volume()) / structure.volume()
            self.assertAlmostEquals(0.0, vol_error, delta=1e-2)
            print "{:d}\t{:.3f}\t{:.2f}".format(structure.n_atoms(),
                    run_time/1000.0, vol_error * 100)

    def test_layered_compound(self):
        structure = VASP5IO.parse_file(
            file_name=self.rel_path+"16234-O2Si1.vasp")

        # Run tessellation.
        result = VoronoiTessellationCalculator.compute(structure,
                                                       radical=False)

        # Test results.
        total_vol = 0.0
        for cell in result:
            total_vol += cell.get_volume()
            self.assertTrue(cell.geometry_is_valid())
        vol_error = (total_vol - structure.volume()) / structure.volume()
        self.assertAlmostEquals(0.0, vol_error, delta=1e-2)