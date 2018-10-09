from __future__ import print_function
from builtins import range
import pandas as pd
import numpy as np
import os
import fnmatch
from rdkit import Chem


def _add_range(file, start_end):
    start = start_end[0::2]
    end = start_end[1::2]
    nit = file.count('%s')
    list_ranges = [
        str(range(start[it], end[it] + 1))[1:-1] for it in range(len(start))
    ]
    s = "pattern = '" + file + "'" + '%(' + str(list_ranges)[1:-1] + ')'
    exec(s)
    return pattern


class RDKitFingerprint(object):
    """ This is an interface to the molecular fingerprints available in RDKit package.

    Parameters
    ----------
    fingerprint_type : str, optional (default='Morgan')
        The type of fingerprint. Available fingerprint types:
            - 'hashed_atom_pair' or 'hap'
            - 'MACCS'
            - 'Morgan'
            - 'hashed_topological_torsion' or 'htt'
            - 'topological_torsion' or 'tt'

    vector: str, optional (default = 'bit')
        Available options for vector:
            - 'int' : represent counts for each fragment instead of bits
                    It is not available for 'MACCS'.
            - 'bit' : only zeros and ones
                    It is not available for 'Topological_torsion'.

    n_bits: int, optional (default = 1024)
        It sets number of elements/bits in the 'bit' type of fingerprint vectors.
        Not availble for:
            - 'MACCS' - (MACCS keys have a fixed length of 167 bits)
            - 'Topological_torsion' - doesn't return a bit vector at all.

    radius: int, optional (default = 2)
        only availble for 'Morgan' fingerprint.

    remove_hydrogen: bool, optional (default=True)
        If True, remove any hydrogen from the graph of a molecule.

    Returns
    -------
    data set of fingerprint vectors.
    """

    def __init__(self,
                 remove_hydrogen=True,
                 fingerprint_type='Morgan',
                 vector='bit',
                 n_bits=1024,
                 radius=2):
        self.remove_hydrogen = remove_hydrogen
        self.fingerprint_type = fingerprint_type
        self.vector = vector
        self.n_bits = n_bits
        self.radius = radius

        if self.vector not in ('bit', 'int'):
            msg = "The parameter vector can be 'int' or 'bit'"
            raise ValueError(msg)

    def read_(self, molfile, path=None, *arguments):
        """ (MolfromFile)
        Construct molecules from one or more input files.

        Parameters
        ----------
        molfile: string
            This is the place you define molecules file name and path. It can contain
            any special character in the following list:

                *       : matches everything
                ?       : matches any single character
                [seq]   : matches any character in seq
                [!seq]  : matches any character not in seq
                /       : filename seperator

            The pattern matching is implemented by fnmatch - Unix filename pattern matching.
            Note: The pattern must include the extension at the end. A list of available
            file extensions is provided here:

                '.mol','.mol2'  : molecule files
                '.pdb'          : protein data bank file
                '.tpl'          : TPL file
                '.smi'          : file of one or more lines of Smiles string
                '.smarts'       : file of one or more lines of Smarts string

        path: string, optional (default = None)
            If path is None, this function tries to open the file as a single
            file (without any pattern matching). Therefore, none of the above
            special characters except '/' are helpful when they are not part
            of the file name or path.
            If not None, it determines the path that this function walk through
            and look for every file that mathes the pattern in the file. To start
            walking from the current directory, the path value should be '.'

        *arguments: integer
            If sequences in the special characters should be a range of integers,
            you can easily pass the start and the end arguments of range as extra
            arguments. But, notice that:
                1- In this case remember to replace seq with %s. Then, you should
                   have [%s] or [!%s] in the file pattern.
                2- for each %s you must pass two integers (always an even number
                    of extra arguments must be passed). Also, the order is important
                    and it should be in the same order of '%s's.

        Pattern Examples
        ----------------
            (1)
            file: 'Mydir/1f/1_opt.smi'
            path: None
            sample files to be read: 'Mydir/1f/1_opt.smi'

            (2)
            file: '[1,2,3,4]?/*_opt.mol'
            path: 'Mydir'
            sample files to be read: 'Mydir/1f/1_opt.mol', 'Mydir/2c/2_opt.mol', ...

            (3)
            file: '[!1,2]?/*_opt.pdb'
            path: '.'
            sample files to be read: './3f/3_opt.pdb', 'Mydir/7c/7_opt.pdb', ...

            (4)
            file: '*['f','c']/*_opt.tpl'
            path: 'Mydir'
            sample files to be read: 'Mydir/1f/1_opt.tpl', 'Mydir/2c/2_opt.tpl', ...

            (5)
            file: '[%s]?/[!%s]_opt.mol2'
            arguments: 1,4,7,10
            path: 'Mydir/all'
            sample files to be read: 'Mydir/all/1f/1_opt.mol2', 'Mydir/all/2c/2_opt.mol2', ...
        """
        extensions = {
            '.mol': Chem.MolFromMolFile,
            '.mol2': Chem.MolFromMol2File,
            '.pdb': Chem.MolFromPDBFile,
            '.tpl': Chem.MolFromTPLFile,
            '.smi': Chem.MolFromSmiles,
            '.smarts': Chem.MolFromSmarts,
            # '.inchi':     Chem.MolFromInchi
        }
        file_name, file_extension = os.path.splitext(molfile)
        if file_extension == '':
            msg = 'The file extension is not determined.'
            raise ValueError(msg)
        elif file_extension not in extensions:
            msg = "The file extension is '%s' not available." % file_extension
            raise ValueError(msg)
        start_end = [arg for arg in arguments]
        if len(start_end) != file_name.count('%s') * 2:
            msg = "Pass an even number of integers, 2 for each '%s'."
            raise ValueError(msg)

        if path and '%s' in molfile:
            molfile = _add_range(molfile, start_end)

        self.removed_rows = []
        self.molecules = []
        if path:
            for root, directories, filenames in os.walk(path):
                file_path = [
                    os.path.join(root, filename) for filename in filenames
                ]
                for filename in sorted(
                        fnmatch.filter(file_path, os.path.join(path,
                                                               molfile))):
                    if file_extension in ['.smi', '.smarts']:
                        mols = open(filename, 'r')
                        mols = mols.readlines()
                        for i, x in enumerate(mols):
                            mol = extensions[file_extension](
                                x.strip())#, removeHs=self.remove_hydrogen)
                            if mol is None:
                                self.removed_rows.append(i)
                            else:
                                self.molecules.append(mol)
                        # mols = [extensions[file_extension](x.strip(),removeHs=self.remove_hydrogen) for x in mols]
                        # self.molecules += mols
                    else:
                        self.molecules += [
                            extensions[file_extension](
                                filename)#, removeHs=self.remove_hydrogen)
                        ]
        else:
            if file_extension in ['.smi', '.smarts']:
                mols = open(molfile, 'r')
                mols = mols.readlines()
                for i, x in enumerate(mols):
                    mol = extensions[file_extension](x.strip())
                    if mol is None:
                        self.removed_rows.append(i)
                    else:
                        self.molecules.append(mol)
                # mols = [extensions[file_extension](x.strip()) for x in mols]
                # self.molecules += mols
            else:
                self.molecules += [
                    extensions[file_extension](
                        file_name, removeHs=self.remove_hydrogen)
                ]

    def fingerprint(self):
        if self.fingerprint_type == 'hashed_atom_pair' or self.fingerprint_type == 'hap':
            if self.vector == 'int':
                from rdkit.Chem.AtomPairs.Pairs import GetHashedAtomPairFingerprint
                self.fps = [
                    GetHashedAtomPairFingerprint(m, nBits=self.n_bits)
                    for m in self.molecules
                ]
                # get nonzero elements as a dictionary for each molecule
                dict_nonzero = [fp.GetNonzeroElements() for fp in self.fps]
                data = pd.DataFrame(dict_nonzero)
                data.fillna(0, inplace=True)
                return data
            elif self.vector == 'bit':
                from rdkit.Chem.rdMolDescriptors import GetHashedAtomPairFingerprintAsBitVect
                self.fps = [
                    GetHashedAtomPairFingerprintAsBitVect(
                        m, nBits=self.n_bits) for m in self.molecules
                ]
                data = np.array(self.fps)
                data = pd.DataFrame(data)
                return data
        elif self.fingerprint_type == 'MACCS':
            if self.vector == 'int':
                msg = "There is no RDKit function to encode integer vectors for MACCS keys"
                raise ValueError(msg)
            elif self.vector == 'bit':
                from rdkit.Chem.MACCSkeys import GenMACCSKeys
                self.fps = [GenMACCSKeys(mol) for mol in self.molecules]
                data = np.array(self.fps)
                data = pd.DataFrame(data)
                return data
        elif self.fingerprint_type == 'Morgan':
            if self.vector == 'int':
                from rdkit.Chem.rdMolDescriptors import GetMorganFingerprint
                self.fps = [
                    GetMorganFingerprint(mol, self.radius)
                    for mol in self.molecules
                ]
                # get nonzero elements as a dictionary for each molecule
                dict_nonzero = [fp.GetNonzeroElements() for fp in self.fps]
                # pairScores = []
                # for fp in dict_nonzero:
                #     pairScores += list(fp)
                data = pd.DataFrame(dict_nonzero)#, columns=list(set(pairScores)))
                data.fillna(0, inplace=True)
                return data
            elif self.vector == 'bit':
                from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
                self.fps = [
                    GetMorganFingerprintAsBitVect(
                        mol, self.radius, nBits=self.n_bits)
                    for mol in self.molecules
                ]
                data = np.array(self.fps)
                data = pd.DataFrame(data)
                return data
        elif self.fingerprint_type == 'hashed_topological_torsion' or self.fingerprint_type == 'htt':
            if self.vector == 'int':
                from rdkit.Chem.rdMolDescriptors import GetHashedTopologicalTorsionFingerprint
                self.fps = [
                    GetHashedTopologicalTorsionFingerprint(
                        m, nBits=self.n_bits) for m in self.molecules
                ]
                # get nonzero elements as a dictionary for each molecule
                dict_nonzero = [fp.GetNonzeroElements() for fp in self.fps]
                data = pd.DataFrame(dict_nonzero)
                data.fillna(0, inplace=True)
                return data
            elif self.vector == 'bit':
                from rdkit.Chem.rdMolDescriptors import GetHashedTopologicalTorsionFingerprintAsBitVect
                self.fps = [
                    GetHashedTopologicalTorsionFingerprintAsBitVect(
                        m, nBits=self.n_bits) for m in self.molecules
                ]
                data = np.array(self.fps)
                data = pd.DataFrame(data)
                return data
        elif self.fingerprint_type == 'topological_torsion' or self.fingerprint_type == 'tt':
            if self.vector == 'int':
                from rdkit.Chem.AtomPairs.Torsions import GetTopologicalTorsionFingerprintAsIntVect
                self.fps = [
                    GetTopologicalTorsionFingerprintAsIntVect(mol)
                    for mol in self.molecules
                ]
                dict_nonzero = [fp.GetNonzeroElements() for fp in self.fps]
                data = pd.DataFrame(dict_nonzero)
                data.fillna(0, inplace=True)
                return data
            elif self.vector == 'bit':
                msg = "There is no RDKit function to encode bit vectors for Topological Torsion Fingerprints"
                raise ValueError(msg)
        else:
            msg = "The type argument '%s' is not a valid fingerprint type" % self.fingerprint_type
            raise ValueError(msg)
