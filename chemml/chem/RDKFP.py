from __future__ import print_function
from builtins import range
import pandas as pd
import numpy as np
import scipy.sparse


from chemml.chem import Molecule

class RDKitFingerprint(object):
    """
    This is an interface to the available molecular fingerprints in the RDKit package.

    Parameters
    ----------
    fingerprint_type : str, optional (default='Morgan')
        The type of fingerprint. Available fingerprint types:
            - 'hashed_atom_pair' or 'hap'
            - 'MACCS' or 'maccs'
            - 'morgan'
            - 'hashed_topological_torsion' or 'htt'
            - 'topological_torsion' or 'tt'

    vector : str, optional (default = 'bit')
        Available options for vector:
            - 'int' : represent counts for each fragment instead of bits
                    It is not available for 'MACCS'.
            - 'bit' : only zeros and ones
                    It is not available for 'Topological_torsion'.

    n_bits : int, optional (default = 1024)
        It sets number of elements/bits in the 'bit' type of fingerprint vectors.
        Not availble for:
            - 'MACCS' - (MACCS keys have a fixed length of 167 bits)
            - 'Topological_torsion' - doesn't return a bit vector at all.

    radius : int, optional (default = 2)
        only applicable if calculating 'Morgan' fingerprint.

    kwargs :
        Any additional argument that should be passed to the rdkit fingerprint function.

    Attributes
    ----------
    n_molecules_ : int
        The number of molecules that are received.

    fps_ : list
        The list of rdkit fingerprint objects.

    """

    def __init__(self,
                 fingerprint_type='Morgan',
                 vector='bit',
                 n_bits=1024,
                 radius=2,
                 **kwargs):
        self.fingerprint_type = fingerprint_type
        self.n_bits = n_bits
        self.radius = radius
        self.kwargs = kwargs
        if not isinstance(vector, str) or vector.lower() not in ('bit', 'int'):
            msg = "The parameter vector must be either 'int' or 'bit'."
            raise ValueError(msg)
        else:
            self.vector = vector.lower()

    def represent(self, molecules):
        """
        The main function to provide fingerprint representation of input molecule(s).

        Parameters
        ----------
        molecules : chemml.chem.Molecule object or list
            It must be an instance of chemml.chem.Molecule object or a list of those objects, otherwise a ValueError will be raised.
            If smiles representation of the molecule (or rdkit molecule object) is not available, we convert the molecule to
            smiles automatically. However, the automatic conversion may ignore your manual settings, for example removed hydrogens,
            kekulized, or canonical smiles.

        Returns
        -------
        features : pandas.DataFrame
            A 2-dimensional pandas dataframe of fingerprint features with same number of rows as number of molecules.

        """
        if isinstance(molecules, list):
            molecules = np.array(molecules)
        elif isinstance(molecules, Molecule):
            molecules = np.array([molecules])
        else:
            msg = "The molecule must be a chemml.chem.Molecule object or a list of objets."
            raise ValueError(msg)

        if molecules.ndim >1:
            msg = "The molecule must be a chemml.chem.Molecule object or a list of objects."
            raise ValueError(msg)

        self.n_molecules_ = molecules.shape[0]

        if self.fingerprint_type.lower() == 'hashed_atom_pair' or self.fingerprint_type.lower() == 'hap':
            return self._hap(molecules)
        elif self.fingerprint_type == 'MACCS' or self.fingerprint_type.lower() == 'maccs':
            return self._maccs(molecules)
        elif self.fingerprint_type.lower() == 'morgan':
            return self._morgan(molecules)
        elif self.fingerprint_type.lower() == 'hashed_topological_torsion' or self.fingerprint_type.lower() == 'htt':
            return self._htt(molecules)
        elif self.fingerprint_type.lower() == 'topological_torsion' or self.fingerprint_type.lower() == 'tt':
            return self._tt(molecules)
        else:
            msg = "The parameter 'fingerprint_type' is not a valid fingerprint type: '%s'" % self.fingerprint_type
            raise ValueError(msg)

    def _hap(self, molecules):
        if self.vector == 'int':
            from rdkit.Chem.AtomPairs.Pairs import GetHashedAtomPairFingerprint
            self.fps_ = [
                GetHashedAtomPairFingerprint(self._sanitary(m), nBits=self.n_bits, **self.kwargs)
                for m in molecules
            ]
            # get nonzero elements as a dictionary for each molecule
            dict_nonzero = [fp.GetNonzeroElements() for fp in self.fps_]
            data = pd.DataFrame(dict_nonzero)
            data.fillna(0, inplace=True)
            return data
        elif self.vector == 'bit':
            from rdkit.Chem.rdMolDescriptors import GetHashedAtomPairFingerprintAsBitVect
            self.fps_ = [
                GetHashedAtomPairFingerprintAsBitVect(
                    self._sanitary(m), nBits=self.n_bits, **self.kwargs) for m in molecules
            ]
            data = np.array(self.fps_)
            data = pd.DataFrame(data)
            return data

    def _maccs(self, molecules):
        if self.vector == 'int':
            msg = "There is no RDKit function to encode integer vectors for MACCS keys"
            raise ValueError(msg)
        elif self.vector == 'bit':
            from rdkit.Chem.MACCSkeys import GenMACCSKeys
            self.fps_ = [GenMACCSKeys(self._sanitary(mol), **self.kwargs) for mol in molecules]
            data = np.array(self.fps_)
            data = pd.DataFrame(data)
            return data

    def _morgan(self, molecules):
        if self.vector == 'int':
            from rdkit.Chem.rdMolDescriptors import GetMorganFingerprint
            self.fps_ = [
                GetMorganFingerprint(self._sanitary(mol), self.radius, **self.kwargs)
                for mol in molecules
            ]
            # get nonzero elements as a dictionary for each molecule
            dict_nonzero = [fp.GetNonzeroElements() for fp in self.fps_]
            # pairScores = []
            # for fp in dict_nonzero:
            #     pairScores += list(fp)
            data = pd.DataFrame(dict_nonzero)#, columns=list(set(pairScores)))
            data.fillna(0, inplace=True)
            return data
        elif self.vector == 'bit':
            from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
            self.fps_ = [
                GetMorganFingerprintAsBitVect(
                    self._sanitary(mol), self.radius, nBits=self.n_bits, **self.kwargs)
                for mol in molecules
            ]
            data = np.array(self.fps_)
            data = pd.DataFrame(data)
            return data

    def _htt(self, molecules):
        if self.vector == 'int':
            from rdkit.Chem.rdMolDescriptors import GetHashedTopologicalTorsionFingerprint
            self.fps_ = [
                GetHashedTopologicalTorsionFingerprint(
                    self._sanitary(mol), nBits=self.n_bits, **self.kwargs) for mol in molecules
            ]
            # get nonzero elements as a dictionary for each molecule
            dict_nonzero = [fp.GetNonzeroElements() for fp in self.fps_]
            data = pd.DataFrame(dict_nonzero)
            data.fillna(0, inplace=True)
            return data
        elif self.vector == 'bit':
            from rdkit.Chem.rdMolDescriptors import GetHashedTopologicalTorsionFingerprintAsBitVect
            self.fps_ = [
                GetHashedTopologicalTorsionFingerprintAsBitVect(
                    self._sanitary(mol), nBits=self.n_bits, **self.kwargs) for mol in molecules
            ]
            data = np.array(self.fps_)
            data = pd.DataFrame(data)
            return data

    def _tt(self, molecules):
        if self.vector == 'int':
            from rdkit.Chem.AtomPairs.Torsions import GetTopologicalTorsionFingerprintAsIntVect
            self.fps_ = [
                GetTopologicalTorsionFingerprintAsIntVect(self._sanitary(mol), **self.kwargs)
                for mol in molecules
            ]
            dict_nonzero = [fp.GetNonzeroElements() for fp in self.fps_]
            data = pd.DataFrame(dict_nonzero)
            data.fillna(0, inplace=True)
            return data
        elif self.vector == 'bit':
            msg = "There is no RDKit function to encode bit vectors for Topological Torsion Fingerprints"
            raise ValueError(msg)

    def _sanitary(self, mol):
        if not isinstance(mol, Molecule):
            msg = "The molecule must be a chemml.chem.Molecule object or a list of objets."
            raise ValueError(msg)
        if mol.rdkit_molecule is None:
            mol.to_smiles()
        return mol.rdkit_molecule

    def store_sparse(self, file, features):
        """
        This function helps you to store higly sparse fingerprint feature sets using `.npz` format for memory efficiency and
        less store/load time.
        Another method of this class, `load_sparse`, enables you to load your `.npz` files and convert it back to pandas dataframe.

        Parameters
        ----------
        file : str
            Must be a path to the file with .npz format.

        features : pandas DataFrame
            Must be the pandas dataframe as you receive it from `represent` method.

        """
        if not isinstance(file, str):
            msg = "The parameter 'file' must be a path to the file with .npz format."
            raise ValueError(msg)

        if not isinstance(features, pd.DataFrame):
            msg = "The parameter 'features' must be a pandas dataframe."
            raise ValueError(msg)

        temp = scipy.sparse.csc_matrix(features.values)
        scipy.sparse.save_npz(file, temp)

    def load_sparse(self, file):
        """
        This function enables you to load sparse matrix with the `.npz` format and convert it to a pandas dataframe.

        Parameters
        ----------
        file : str
            Must be a path to the file with .npz format.

        Returns
        -------
        features : pandas.DataFrame
            The dense dataframe of the passed sparse file.

        """
        if not isinstance(file, str):
            msg = "The parameter 'file' must be a path to the file with .npz format."
            raise ValueError(msg)

        temp = scipy.sparse.load_npz(file)
        return pd.DataFrame(temp.todense())
