from __future__ import print_function
from builtins import range
import os
import pandas as pd
import numpy as np
import warnings
import fnmatch

from chemml.chem.molecule import Molecule

class Split(object):
    """
    split data frame by columns

    Parameters
    ----------
    select: int or list, optional (default = 1)
        if integer, it's the number of columns to be selected from left side of data and returns as first data frame (X1).
        if list, it must contain the exact column names to be selected and returned as first data frame (X1).
        The remaining columns will be returned as second dataframe (X2) in both cases.

    """

    def __init__(self, selection=1):
        self.selection = selection

    def fit(self, X):
        """
        The main function to fit the split task to the input data frame X.

        Parameters
        ----------
        X: pandas dataframe
            The input dataframe

        Returns
        -------
        pandas dataframe
            X1: the dataframe resulted based on the selection parameter
        pandas dataframe
            X2: the dataframe of columns not selected as X1
        """
        if not isinstance(X, pd.DataFrame):
            msg = 'X must be a pandas dataframe'
            raise ValueError(msg)
        if isinstance(self.selection, list):
            X1 = X.loc[:, self.selection]
            X2 = X.drop(self.selection, axis=1)
        elif isinstance(self.selection, int):
            if self.selection >= X.shape[1]:
                msg = 'The first output data frame is empty, because passed a bigger number than actual number of columns'
                warnings.warn(msg)
            X1 = X.iloc[:, :self.selection]
            X2 = None
        else:
            msg = "The parameter selection must ba a list or an integer"
            raise ValueError(msg)
        return X1, X2


class XYZreader(object):
    """ (XYZreader)
    Read molecules' geometry (cartesian coordinates) from one or more XYZ files.

    Parameters
    ----------
    path_pattern: string or list of string
        A pattern or a list of patterns. Each pattern consists of file name and path.
        The pattern can contain any special character in the following list:

            *       : matches everything (zero or more characters)
            ?       : matches any single character
            [seq]   : matches any character in seq
            [!seq]  : matches any character not in seq
            /       : filename seperator

        Note: seq can indicate a range of characters by giving two characters and separating them by a '-'.
              However, the range is limited to a single character. That's why this parameter also accept a list of
              patterns for different length of characters. For example, range(1,30) = '[1-9]' and '[1-9][0-9]'

        The pattern matching is utilizes fnmatch library- Unix filename pattern matching.
        Note: The pattern must include the file format at the end. The only acceptable extension is '.xyz'.

    path_root: string, optional (default = None)
        fixed (with no special character) part of the path.
        If path is None, this function tries to open the file as a single
        file (without any pattern matching). It is a fixed path to subdirectories
        or files. Therefore, none of the above special characters except '/'
        can be used.
        If not None, it determines the path that this function walk through
        and look for every file that matches the pattern. To start
        walking from the curent directory, the path should be '.'

    Z: dictionary, optional (default = {'Ru': 44.0, 'Re': 75.0, 'Rf': 104.0, 'Rg': 111.0, 'Ra': 88.0, 'Rb': 37.0, 'Rn': 86.0, 'Rh': 45.0, 'Be': 4.0, 'Ba': 56.0, 'Bh': 107.0, 'Bi': 83.0, 'Bk': 97.0, 'Br': 35.0, 'H': 1.0, 'P': 15.0, 'Os': 76.0, 'Es': 99.0, 'Hg': 80.0, 'Ge': 32.0, 'Gd': 64.0, 'Ga': 31.0, 'Pr': 59.0, 'Pt': 78.0, 'Pu': 94.0, 'C': 6.0, 'Pb': 82.0, 'Pa': 91.0, 'Pd': 46.0, 'Cd': 48.0, 'Po': 84.0, 'Pm': 61.0, 'Hs': 108.0, 'Uup': 115.0, 'Uus': 117.0, 'Uuo': 118.0, 'Ho': 67.0, 'Hf': 72.0, 'K': 19.0, 'He': 2.0, 'Md': 101.0, 'Mg': 12.0, 'Mo': 42.0, 'Mn': 25.0, 'O': 8.0, 'Mt': 109.0, 'S': 16.0, 'W': 74.0, 'Zn': 30.0, 'Eu': 63.0, 'Zr': 40.0, 'Er': 68.0, 'Ni': 28.0, 'No': 102.0, 'Na': 11.0, 'Nb': 41.0, 'Nd': 60.0, 'Ne': 10.0, 'Np': 93.0, 'Fr': 87.0, 'Fe': 26.0, 'Fl': 114.0, 'Fm': 100.0, 'B': 5.0, 'F': 9.0, 'Sr': 38.0, 'N': 7.0, 'Kr': 36.0, 'Si': 14.0, 'Sn': 50.0, 'Sm': 62.0, 'V': 23.0, 'Sc': 21.0, 'Sb': 51.0, 'Sg': 106.0, 'Se': 34.0, 'Co': 27.0, 'Cn': 112.0, 'Cm': 96.0, 'Cl': 17.0, 'Ca': 20.0, 'Cf': 98.0, 'Ce': 58.0, 'Xe': 54.0, 'Lu': 71.0, 'Cs': 55.0, 'Cr': 24.0, 'Cu': 29.0, 'La': 57.0, 'Li': 3.0, 'Lv': 116.0, 'Tl': 81.0, 'Tm': 69.0, 'Lr': 103.0, 'Th': 90.0, 'Ti': 22.0, 'Te': 52.0, 'Tb': 65.0, 'Tc': 43.0, 'Ta': 73.0, 'Yb': 70.0, 'Db': 105.0, 'Dy': 66.0, 'Ds': 110.0, 'I': 53.0, 'U': 92.0, 'Y': 39.0, 'Ac': 89.0, 'Ag': 47.0, 'Uut': 113.0, 'Ir': 77.0, 'Am': 95.0, 'Al': 13.0, 'As': 33.0, 'Ar': 18.0, 'Au': 79.0, 'At': 85.0, 'In': 49.0})
        A dictionary of nuclear charges with respect to the chemical symbols of all atom types in the xyz files.

    reader: string, optional (default = 'auto')
        Available options : 'auto' and 'manual'
        If 'auto', the openbabel readstring function creat the molecule object.
        The type of files for openbabel class has been set to 'xyz', thus the format
        of the file should also follow a typical xyz format. However, with 'manual'
        reader you can skip some lines from top or bottom of xyz files.

    skip_lines: list of two integers, optional (default = [2,0])
        Number of lines to skip (int) from top and bottom of the xyz files, respectively.
        Based on the original xyz format only 2 lines from top and zero from
        bottom of a file can be skipped. Thus, number of atoms in the first line
        can also be ignored.
        Only available for 'manual' reader.

    Attributes
    ----------
    max_n_atoms_: int
        Maximum number of atoms in the pool of molecules.
        This can be useful if you want to set this parameter in the feature representation methods,
        e.g. Coulomb_Matrix.

    Notes
    --------
        Some pattern examples:
        (1)
        path_pattern: 'Mydir/1f/1_opt.xyz'
        path_root: None
        one file will be read: 'Mydir/1f/1_opt.xyz'

        (2)
        path_pattern: '[1,2,3,4]?/*_opt.xyz'
        path_root: 'Mydir'
        sample files to be read: 'Mydir/1f/1_opt.xyz', 'Mydir/2c/2_opt.xyz', ...

        (3)
        path_pattern: '[!1,2]?/*_opt.xyz'
        path_root: '.'
        sample files to be read: './3f/3_opt.xyz', 'Mydir/7c/7_opt.xyz', ...

        (4)
        path_pattern: '*['f','c']/*_opt.xyz'
        path_root: 'Mydir'
        sample files to be read: 'Mydir/1f/1_opt.xyz', 'Mydir/2c/2_opt.xyz', ...

        (5)
        path_pattern: ['[2-5]_opt.xyz', '[1-9][2-5]_opt.xyz']
        path_root: 'Mydir/all'
        sample files to be read: 'Mydir/all/1f/1_opt.xyz', 'Mydir/all/2c/2_opt.xyz', ...
    """

    def __init__(
            self,
            path_pattern,
            path_root=None,
            Z={
                'Ru': 44.0,
                'Re': 75.0,
                'Rf': 104.0,
                'Rg': 111.0,
                'Ra': 88.0,
                'Rb': 37.0,
                'Rn': 86.0,
                'Rh': 45.0,
                'Be': 4.0,
                'Ba': 56.0,
                'Bh': 107.0,
                'Bi': 83.0,
                'Bk': 97.0,
                'Br': 35.0,
                'H': 1.0,
                'P': 15.0,
                'Os': 76.0,
                'Es': 99.0,
                'Hg': 80.0,
                'Ge': 32.0,
                'Gd': 64.0,
                'Ga': 31.0,
                'Pr': 59.0,
                'Pt': 78.0,
                'Pu': 94.0,
                'C': 6.0,
                'Pb': 82.0,
                'Pa': 91.0,
                'Pd': 46.0,
                'Cd': 48.0,
                'Po': 84.0,
                'Pm': 61.0,
                'Hs': 108.0,
                'Uup': 115.0,
                'Uus': 117.0,
                'Uuo': 118.0,
                'Ho': 67.0,
                'Hf': 72.0,
                'K': 19.0,
                'He': 2.0,
                'Md': 101.0,
                'Mg': 12.0,
                'Mo': 42.0,
                'Mn': 25.0,
                'O': 8.0,
                'Mt': 109.0,
                'S': 16.0,
                'W': 74.0,
                'Zn': 30.0,
                'Eu': 63.0,
                'Zr': 40.0,
                'Er': 68.0,
                'Ni': 28.0,
                'No': 102.0,
                'Na': 11.0,
                'Nb': 41.0,
                'Nd': 60.0,
                'Ne': 10.0,
                'Np': 93.0,
                'Fr': 87.0,
                'Fe': 26.0,
                'Fl': 114.0,
                'Fm': 100.0,
                'B': 5.0,
                'F': 9.0,
                'Sr': 38.0,
                'N': 7.0,
                'Kr': 36.0,
                'Si': 14.0,
                'Sn': 50.0,
                'Sm': 62.0,
                'V': 23.0,
                'Sc': 21.0,
                'Sb': 51.0,
                'Sg': 106.0,
                'Se': 34.0,
                'Co': 27.0,
                'Cn': 112.0,
                'Cm': 96.0,
                'Cl': 17.0,
                'Ca': 20.0,
                'Cf': 98.0,
                'Ce': 58.0,
                'Xe': 54.0,
                'Lu': 71.0,
                'Cs': 55.0,
                'Cr': 24.0,
                'Cu': 29.0,
                'La': 57.0,
                'Li': 3.0,
                'Lv': 116.0,
                'Tl': 81.0,
                'Tm': 69.0,
                'Lr': 103.0,
                'Th': 90.0,
                'Ti': 22.0,
                'Te': 52.0,
                'Tb': 65.0,
                'Tc': 43.0,
                'Ta': 73.0,
                'Yb': 70.0,
                'Db': 105.0,
                'Dy': 66.0,
                'Ds': 110.0,
                'I': 53.0,
                'U': 92.0,
                'Y': 39.0,
                'Ac': 89.0,
                'Ag': 47.0,
                'Uut': 113.0,
                'Ir': 77.0,
                'Am': 95.0,
                'Al': 13.0,
                'As': 33.0,
                'Ar': 18.0,
                'Au': 79.0,
                'At': 85.0,
                'In': 49.0
            },
            reader='auto',
            skip_lines=[2, 0],
            path_only=False):
        self.path_pattern = path_pattern
        self.path_root = path_root
        self.Z = Z
        self.reader = reader
        self.skip_lines = skip_lines
        self.path_only = path_only

    def __file_reader(self, filename):
        if self.reader == 'auto':
            # sys.path.insert(0, "/user/m27/pkg/openbabel/2.3.2/lib")
            from openbabel import pybel
            # import openbabel
            mol = open(filename, 'r').read()
            mol = pybel.readstring("xyz", mol)
            molecule = [(a.OBAtom.GetAtomicNum(), a.OBAtom.x(), a.OBAtom.y(),
                         a.OBAtom.z()) for a in mol.atoms]
            return np.array(molecule)
        elif self.reader == 'manual':
            mol = open(filename, 'r').readlines()
            if len(mol) == 0:
                return np.array([])
            mol = mol[self.skip_lines[0]:len(mol) - self.skip_lines[1]]
            molecule = []
            for atom in mol:
                atom = atom.replace('\t', ' ')
                atom = atom.strip().split(' ')
                atom = list(filter(lambda x: x != '', atom))
                molecule.append([
                    self.Z[atom[0]],
                    float(atom[1]),
                    float(atom[2]),
                    float(atom[3])
                ])
            return np.array(molecule)

    def read(self):
        """
        read the XYZ files based on the path_pattern and path_root parameters and create a list of chemml.chem.Molecule objects.

        Return
        ------
        molecules: list
            A list of chemml.chem.Molecule objects
        """
        if isinstance(self.path_pattern, str):
            self.path_pattern = [self.path_pattern]
        molecules = {}
        it = 0
        max_nAtoms = 1
        for pattern in self.path_pattern:
            file_name, file_extension = os.path.splitext(pattern)
            if file_extension == '':
                msg = 'The file extension must be indicated.'
                raise ValueError(msg)
            elif file_extension != '.xyz':
                msg = "file extension '%s' not available - xyz is the only acceptable extension" % file_extension
                raise ValueError(msg)

            if self.path_root:
                for root, directories, filenames in os.walk(self.path_root):
                    file_path = [
                        os.path.join(root, filename) for filename in filenames
                    ]
                    for fn in sorted(
                            fnmatch.filter(
                                file_path, os.path.join(
                                    self.path_root, pattern))):
                        if not self.path_only:
                            mol = self.__file_reader(fn)
                            max_nAtoms = max(max_nAtoms, len(mol))
                        else:
                            mol = None
                            max_nAtoms = max(max_nAtoms, 0)
                        it += 1
                        molecules[it] = {'file': fn, 'mol': mol}
            else:
                if not self.path_only:
                    mol = self.__file_reader(file_name + file_extension)
                    max_nAtoms = max(max_nAtoms, len(mol))
                else:
                    mol = None
                    max_nAtoms = max(max_nAtoms, 0)
                it += 1
                molecules[it] = {'file': file_name, 'mol': mol}
        self.max_n_atoms_ = max_nAtoms
        return molecules


