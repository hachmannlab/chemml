import pandas as pd
import numpy as np
import scipy.sparse
from tqdm import tqdm

from rdkit.Chem import rdFingerprintGenerator

from chemml.chem import Molecule
from chemml.chem.foss_descriptors import normalize_input

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

    n_bits : int, optional (default = 1024)
        It sets number of elements/bits in the 'bit' type of fingerprint vectors.
        Not available for:
            - 'MACCS' - (MACCS keys have a fixed length of 167 bits)

    radius : int, optional (default = 2)
        only applicable if calculating 'Morgan' fingerprint.

    kwargs :
        Any additional argument that should be passed to the rdkit fingerprint generator.
        For backward compatibility, the following old parameter names are mapped:
            - Morgan: 'useChirality' -> 'includeChirality', 'useFeatures' -> uses
              GetMorganFeatureAtomInvGen(), 'useBondTypes' passed through
            - AtomPair: 'minLength' -> 'minDistance', 'maxLength' -> 'maxDistance'

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
        _, _, molecules = normalize_input(molecules, force_molecule=True)

        self.n_molecules_ = len(molecules)

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

    def _map_morgan_kwargs(self):
        """Map old Morgan kwargs to new rdFingerprintGenerator params."""
        mapped = {}
        kwargs = dict(self.kwargs)
        if 'useChirality' in kwargs:
            mapped['includeChirality'] = kwargs.pop('useChirality')
        if 'useFeatures' in kwargs:
            if kwargs.pop('useFeatures'):
                mapped['atomInvariantsGenerator'] = rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
        if 'useBondTypes' in kwargs:
            mapped['useBondTypes'] = kwargs.pop('useBondTypes')
        mapped.update(kwargs)
        return mapped

    def _map_atompair_kwargs(self):
        """Map old AtomPair kwargs to new rdFingerprintGenerator params."""
        mapped = {}
        kwargs = dict(self.kwargs)
        if 'minLength' in kwargs:
            mapped['minDistance'] = kwargs.pop('minLength')
        if 'maxLength' in kwargs:
            mapped['maxDistance'] = kwargs.pop('maxLength')
        mapped.update(kwargs)
        return mapped

    def _map_tt_kwargs(self):
        """Map old TopologicalTorsion kwargs to new rdFingerprintGenerator params."""
        return dict(self.kwargs)

    def _hap(self, molecules):
        mapped_kwargs = self._map_atompair_kwargs()
        gen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=self.n_bits, **mapped_kwargs)
        if self.vector == 'int':
            self.fps_ = [gen.GetCountFingerprint(self._sanitary(m)) for m in tqdm(molecules, desc="Generating Atom Pair Fingerprints", unit="molecule")]
            dict_nonzero = [fp.GetNonzeroElements() for fp in self.fps_]
            data = pd.DataFrame(dict_nonzero)
            data.fillna(0, inplace=True)
            return data
        elif self.vector == 'bit':
            self.fps_ = [gen.GetFingerprint(self._sanitary(m)) for m in tqdm(molecules, desc="Generating Atom Pair Fingerprints", unit="molecule")]
            data = np.array(self.fps_)
            data = pd.DataFrame(data)
            return data

    def _maccs(self, molecules):
        if self.vector == 'int':
            msg = "There is no RDKit function to encode integer vectors for MACCS keys"
            raise ValueError(msg)
        elif self.vector == 'bit':
            from rdkit.Chem.MACCSkeys import GenMACCSKeys
            self.fps_ = [GenMACCSKeys(self._sanitary(mol), **self.kwargs) for mol in tqdm(molecules, desc="Generating MACCS Fingerprints", unit="molecule")]
            data = np.array(self.fps_)
            data = pd.DataFrame(data)
            return data

    def _morgan(self, molecules):
        mapped_kwargs = self._map_morgan_kwargs()
        if self.vector == 'int':
            gen = rdFingerprintGenerator.GetMorganGenerator(radius=self.radius, **mapped_kwargs)
            self.fps_ = [gen.GetSparseCountFingerprint(self._sanitary(mol)) for mol in tqdm(molecules, desc="Generating Morgan Fingerprints", unit="molecule")]
            dict_nonzero = [fp.GetNonzeroElements() for fp in self.fps_]
            data = pd.DataFrame(dict_nonzero)
            data.fillna(0, inplace=True)
            return data
        elif self.vector == 'bit':
            gen = rdFingerprintGenerator.GetMorganGenerator(
                radius=self.radius, fpSize=self.n_bits, **mapped_kwargs)
            self.fps_ = [gen.GetFingerprint(self._sanitary(mol)) for mol in tqdm(molecules, desc="Generating Morgan Fingerprints", unit="molecule")]
            data = np.array(self.fps_)
            data = pd.DataFrame(data)
            return data

    def _htt(self, molecules):
        mapped_kwargs = self._map_tt_kwargs()
        gen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=self.n_bits, **mapped_kwargs)
        if self.vector == 'int':
            self.fps_ = [gen.GetCountFingerprint(self._sanitary(mol)) for mol in tqdm(molecules, desc="Generating Hashed Topological Torsion Fingerprints", unit="molecule")]
            dict_nonzero = [fp.GetNonzeroElements() for fp in self.fps_]
            data = pd.DataFrame(dict_nonzero)
            data.fillna(0, inplace=True)
            return data
        elif self.vector == 'bit':
            self.fps_ = [gen.GetFingerprint(self._sanitary(mol)) for mol in tqdm(molecules, desc="Generating Hashed Topological Torsion Fingerprints", unit="molecule")]
            data = np.array(self.fps_)
            data = pd.DataFrame(data)
            return data

    def _tt(self, molecules):
        mapped_kwargs = self._map_tt_kwargs()
        if self.vector == 'int':
            gen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(**mapped_kwargs)
            self.fps_ = [gen.GetSparseCountFingerprint(self._sanitary(mol)) for mol in tqdm(molecules, desc="Generating Topological Torsion Fingerprints", unit="molecule")]
            dict_nonzero = [fp.GetNonzeroElements() for fp in self.fps_]
            data = pd.DataFrame(dict_nonzero)
            data.fillna(0, inplace=True)
            return data
        elif self.vector == 'bit':
            gen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=self.n_bits, **mapped_kwargs)
            self.fps_ = [gen.GetFingerprint(self._sanitary(mol)) for mol in tqdm(molecules, desc="Generating Topological Torsion Fingerprints", unit="molecule")]
            data = np.array(self.fps_)
            data = pd.DataFrame(data)
            return data

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
