from collections import OrderedDict
import numpy as np
from ..data.Atom import Atom
from ..data.Cell import Cell

class VASP5IO:
    """Class to read/write from VASP 5 format. Note that this version expects
    that the sixth line to contain the names of elements.

    """

    @classmethod
    def parse_file(self, file_name=None, list_of_lines=None):
        """Function to read a structure from a file or list of lines.

        Parameters
        ----------
        file_name : str
            Path to file. (Default value = None)
        list_of_lines : array-like
            Lines describing structure. (Default value = None)

        Returns
        -------
        output : Cell
            Structure as Cell.

        Raises
        ------
        Exception
            If something is wrong with input.
            If there is an error parsing the basis.
            If there is an error parsing atom types.
            If there is an error determining whether atoms are in cartesian
            units.
        """

        # Open file.
        lines = None
        if list_of_lines is None:
            with open(file_name, 'r') as f:
                # Add contents to file.
                lines = [line.strip() for line in f.readlines()]
        else:
            lines = [l.strip() for l in list_of_lines]

        if lines is None:
            raise Exception("Something went wrong. Check input.")

        # Get basis.
        factor = float(lines[1])
        basis = np.zeros((3, 3))
        try:
            for i in range(3):
                words = lines[2 + i].split()
                for j in range(3):
                    # VASP's basis is transposed so that each row is a
                    # lattice vector. See:
                    # http://cms.mpi.univie.ac.at/vasp/guide/node59.html
                    basis[j][i] = float(words[j]) * factor
        except Exception:
            raise Exception("Error parsing basis.")

        # Read atom types.
        types = None
        type_count = None
        try:
            types = lines[5].split()
            words = lines[6].split()
            type_count = [int(words[i]) for i in range(len(types))]
        except Exception:
            raise Exception("Error parsing atom types.")

        # Read middle section.
        atom_start = None
        cartesian = None
        try:
            # Check whether SD is on the see where atom positions start.
            atom_start = 7 if lines[7].lower().startswith("sele") else 8

            # See whether coordinates are in direct or cartesian units.
            cartesian = lines[atom_start - 1].lower().startswith("c")
        except Exception:
            raise Exception("Error determining whether atoms are in cartesian "
                            "units.")

        # Make the cell.
        structure = Cell()
        structure.set_basis(basis=basis)

        # Get atom positions.
        try:
            for t in range(len(types)):
                for ti in range(type_count[t]):
                    # Read position.
                    x = [float(w) for w in lines[atom_start].split()]
                    atom_start += 1
                    if cartesian:
                        x = structure.convert_cartesian_to_fractional(x)
                    atom = Atom(x, t)
                    structure.add_atom(atom)
                structure.set_type_name(t, types[t])
        except Exception:
            raise Exception("Error parsing atoms.")

        return structure

    def convert_structure_to_string(self, structure):
        """Function to convert structure to string representation - list of
        lines.

        Parameters
        ----------
        structure : Cell
            Desired structure to be converted.

        Returns
        -------
        output : array-like
            String representation - list of lines.

        """
        output = []

        # Write header.
        output.append("Automatically-generated POSCAR")
        output.append("1.0")

        # Write lattice vectors.
        lat_vectors = structure.get_lattice_vectors()
        for i in range(3):
            output.append(" {0:.10f} {1:.10f} {2:.10f}".format(lat_vectors[
                i][0], lat_vectors[i][1], lat_vectors[i][2]))

        # Gather atoms.
        names = OrderedDict()
        atoms = OrderedDict()
        for atom in structure.get_atoms():
            type = atom.get_type()
            if type in names:
                atoms[type].append(atom)
            else:
                names[type] = structure.get_type_name(type)
                atoms[type] = [atom]

        # Print atom types.
        name_line = ""
        count_line = ""
        for key in names:
            name_line += " "+str(key)
            count_line += " "+str(len(atoms[key]))

        output.append(name_line)
        output.append(count_line)

        output.append("Direct")

        # Print atoms.
        for key in atoms:
            for atom in atoms[key]:
                x = atom.get_position()
                output.append("{0:.10f} {1:.10f} {2:.10f}".format(x[0], x[1],
                                                                  x[2]))
        return output

    def write_structure_to_file(self, structure, filename):
        """Function to write a structure to a file.

        Parameters
        ----------
        structure : Cell
            Desired structure as Cell.
        filename : str
            Desired file name.

        """

        # Generate description.
        to_write = self.convert_structure_to_string(structure)

        # Write to file.
        with open(filename, 'w') as f:
            f.writelines(to_write)

    def print_structure(self, structure):
        """Function to print the structure to standard out.

        Parameters
        ----------
        structure : Cell
            Desired structure as Cell.

        Returns
        -------
        output : array-like
            String representation - list of lines.

        """

        # Get the structure as a list of strings.
        lines = self.convert_structure_to_string(structure)
        output = "\n".join(lines)
        return output