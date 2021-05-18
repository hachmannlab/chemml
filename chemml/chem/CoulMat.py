from builtins import range
import pandas as pd
import numpy as np

from functools import partial
from multiprocessing import cpu_count, Pool

from tensorflow.keras.utils import Progbar

from chemml.chem import Molecule
from chemml.utils import padaxis


class CoulombMatrix(object):
    """
    The implementation of coulomb matrix descriptors by Matthias Rupp et. al. 2012, PRL (All 3 different variations).

    Parameters
    ----------
    cm_type : str, optional (default='SC')
        The coulomb matrix type, one of the following types:
            * 'Unsorted_Matrix' or 'UM'
            * 'Unsorted_Triangular' or 'UT'
            * 'Eigenspectrum' or 'E' 
            * 'Sorted_Coulomb' or 'SC'
            * 'Random_Coulomb' or 'RC'

    max_n_atoms : int or 'auto', optional (default = 'auto')
        Set the maximum number of atoms per molecule (to which all representations will be padded).
        If 'auto', we find it based on all input molecules.

    nPerm : int, optional (default = 3)
        Number of permutation of coulomb matrix per molecule for Random_Coulomb (RC) 
        type of representation.

    const : float, optional (default = 1)
            The constant value for coordinates unit conversion to atomic unit
            example: atomic unit -> const=1, Angstrom -> const=0.529
            const/|Ri-Rj|, which denominator is the euclidean distance
            between atoms i and j

    n_jobs : int, optional(default=-1)
        The number of parallel processes. If -1, uses all the available processes.

    verbose : bool, optional(default=True)
        The verbosity of messages.

    Attributes
    ----------
    n_molecules_ : int
        Total number of molecules.

    max_n_atoms_ : int
        Maximum number of atoms in all molecules.

    Examples
    --------
    >>> from chemml.chem import CoulombMatrix, Molecule

    >>> m1 = Molecule('c1ccc1', 'smiles')
    >>> m2 = Molecule('CNC', 'smiles')
    >>> m3 = Molecule('CC', 'smiles')
    >>> m4 = Molecule('CCC', 'smiles')

    >>> molecules = [m1, m2, m3, m4]

    >>> for mol in molecules:   mol.to_xyz(optimizer='UFF')
    >>> cm = CoulombMatrix(cm_type='SC', n_jobs=-1)
    >>> features = cm.represent(molecules)
    """
    def __init__(self, cm_type='SC', max_n_atoms = 'auto', nPerm=3, const=1,
                 n_jobs=-1, verbose=True):
        self.CMtype = cm_type
        self.max_n_atoms_ = max_n_atoms
        self.nPerm = nPerm
        self.const = const
        self.n_jobs = n_jobs
        self.verbose = verbose

    def __cal_coul_mat(self, mol):
        """

        Parameters
        ----------
        mol: molecule object


        Returns
        -------

        """
        if isinstance(mol, Molecule):
            if mol.xyz is None:
                msg = "The molecule must be a chemml.chem.Molecule object with xyz information."
                raise ValueError(msg)
        else:
            msg = "The molecule must be a chemml.chem.Molecule object."
            raise ValueError(msg)

        mol = np.append(mol.xyz.atomic_numbers,mol.xyz.geometry, axis=1)
        cm = []
        for i in range(len(mol)):
            vect = []
            for k in range(0,i):
                vect.append(cm[k][i])
            for j in range(i,len(mol)):
                if i==j:
                    vect.append(0.5*mol[i,0]**2.4)
                else:
                    vect.append((mol[i,0]*mol[j,0]*self.const)/np.linalg.norm(mol[i,1:]-mol[j,1:]))
            for m in range(len(mol), self.max_n_atoms_):
                vect.append(0.0)
            cm.append(vect)

        # pad with zero values
        if self.max_n_atoms_ > len(mol):
            cm = padaxis(np.array(cm), self.max_n_atoms_, 0, 0)
        return np.array(cm)[:self.max_n_atoms_, :self.max_n_atoms_] #shape nAtoms*nAtoms

    def represent(self, molecules):
        """
        provides coulomb matrix representation for input molecules.

        Parameters
        ----------
        molecules : chemml.chem.Molecule object or array
            If list, it must be a list of chemml.chem.Molecule objects, otherwise we raise a ValueError.
            In addition, all the molecule objects must provide the XYZ information. Please make sure the XYZ geometry has been
            stored or optimized in advance.

        Returns
        -------
        features : Pandas DataFrame
            A data frame with same number of rows as number of molecules will be returned.
            The exact shape of the dataframe depends on the type of CM as follows:
                - shape of Unsorted_Matrix (UM): (n_molecules, max_n_atoms**2)
                - shape of Unsorted_Triangular (UT): (n_molecules, max_n_atoms*(max_n_atoms+1)/2)
                - shape of eigenspectrums (E): (n_molecules, max_n_atoms)
                - shape of Sorted_Coulomb (SC): (n_molecules, max_n_atoms*(max_n_atoms+1)/2)
                - shape of Random_Coulomb (RC): (n_molecules, nPerm * max_n_atoms * (max_n_atoms+1)/2)
        """
        # check input molecules
        if isinstance(molecules, (list,np.ndarray)):
            molecules = np.array(molecules)
        elif isinstance(molecules, Molecule):
            molecules = np.array([molecules])
        else:
            msg = "The molecule must be a chemml.chem.Molecule object or a list of objects."
            raise ValueError(msg)

        if molecules.ndim >1:
            msg = "The molecule must be a chemml.chem.Molecule object or a list of objects."
            raise ValueError(msg)

        self.n_molecules_ = molecules.shape[0]

        # max number of atoms based on the list of molecules
        if self.max_n_atoms_ == 'auto':
            try:
                self.max_n_atoms_ = max([m.xyz.atomic_numbers.shape[0] for m in molecules])
            except:
                msg = "The xyz representation of molecules is not available."
                raise ValueError(msg)

        # pool of processes
        if self.n_jobs == -1:
            self.n_jobs = cpu_count()
        pool = Pool(processes=self.n_jobs)

        # Create an iterator
        # http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
        def chunks(l, n):
            """Yield successive n-sized chunks from l."""
            for i in range(0, len(l), n):
                yield l[i:i + n]

        # find size of each batch
        batch_size = int(len(molecules) / self.n_jobs)
        if batch_size == 0:
            batch_size = 1

        molecule_chunks = chunks(molecules, batch_size)

        # MAP: CM in parallel
        map_function = partial(self._represent)
        if self.verbose:
            print('featurizing molecules in batches of %i ...' % batch_size)
            pbar = Progbar(len(molecules), width=50)
            tensor_list = []
            for tensors in pool.imap(map_function, molecule_chunks):
                pbar.add(len(tensors[0]))
                tensor_list.append(tensors)
            print('Merging batch features ...    ', end='')
        else:
            tensor_list = pool.map(map_function, molecule_chunks)
        if self.verbose:
            print('[DONE]')

        # REDUCE: Concatenate the obtained tensors
        pool.close()
        pool.join()
        return pd.concat(tensor_list, axis=0, ignore_index=True)

    def _represent(self, molecules):

        # in parallel run the number of molecules is different from self.n_molecules_
        n_molecules_ = len(molecules)

        if self.CMtype == "Unsorted_Matrix" or self.CMtype == 'UM':
            cms = np.array([])
            for mol in molecules:
                cm = self.__cal_coul_mat(mol)
                cms = np.append(cms, cm.ravel())
            cms = cms.reshape(n_molecules_, self.max_n_atoms_ ** 2)
            cms = pd.DataFrame(cms)
            return cms

        elif self.CMtype == "Unsorted_Triangular" or self.CMtype == 'UT':
            cms = np.array([])
            for mol in molecules:
                cm = self.__cal_coul_mat(mol)
                cms = np.append(cms, cm[np.tril_indices(self.max_n_atoms_)])
            cms = cms.reshape(n_molecules_, int(self.max_n_atoms_ * (self.max_n_atoms_ + 1) / 2))
            cms = pd.DataFrame(cms)
            return cms

        elif self.CMtype == 'Eigenspectrum' or self.CMtype == 'E':
            eigenspectrums = np.array([])
            for mol in molecules:
                cm = self.__cal_coul_mat(mol) # Check the constant value for unit conversion; atomic unit -> 1 , Angstrom -> 0.529
                eig = np.linalg.eigvals(cm)
                eig[::-1].sort()
                eigenspectrums = np.append(eigenspectrums,eig)
            eigenspectrums = eigenspectrums.reshape(n_molecules_, self.max_n_atoms_)
            eigenspectrums = pd.DataFrame(eigenspectrums)
            return eigenspectrums
            
        elif self.CMtype == 'Sorted_Coulomb' or self.CMtype == 'SC':
            sorted_cm = np.array([])
            for mol in molecules:
                cm = self.__cal_coul_mat(mol)
                lambdas = np.linalg.norm(cm,2,1)
                sort_indices = np.argsort(lambdas)[::-1]
                cm = cm[:,sort_indices][sort_indices,:]
                # sorted_cm.append(cm)
                sorted_cm = np.append(sorted_cm, cm[np.tril_indices(self.max_n_atoms_)]) # lower-triangular
            sorted_cm = sorted_cm.reshape(n_molecules_, int(self.max_n_atoms_ * (self.max_n_atoms_ + 1) / 2))
            sorted_cm = pd.DataFrame(sorted_cm)
            return sorted_cm
                    
        elif self.CMtype == 'Random_Coulomb' or self.CMtype == 'RC':
            random_cm = np.array([])
            for nmol,mol in enumerate(molecules):
                cm = self.__cal_coul_mat(mol)
                lambdas = np.linalg.norm(cm,2,1)
                sort_indices = np.argsort(lambdas)[::-1]
                cm = cm[:,sort_indices][sort_indices,:]
                cm_perms = []
                for l in range(self.nPerm):
                    mask = np.random.permutation(self.max_n_atoms_)
                    cm_mask = cm[:,mask][mask,:]
                    # cm_perms.append(cm_mask)
                    cm_perms += list(cm_mask[np.tril_indices(self.max_n_atoms_)]) # lower-triangular
                random_cm = np.append(random_cm, cm_perms)
            random_cm = random_cm.reshape(n_molecules_, int(self.nPerm * self.max_n_atoms_ * (self.max_n_atoms_ + 1) / 2))
            return pd.DataFrame(random_cm)

    @staticmethod
    def concat_dataframes(mol_tensors_list):
        """
        Concatenates a list of molecule tensors

        Parameters
        ----------
            mol_tensors_list: list
                list of molecule tensors

        Returns
        -------
        features : dataframe
            a single feature dataframe
        """

        assert isinstance(mol_tensors_list, (tuple, list)), 'Provide a list or tuple of molecule tensors to concatenate'

        #stack along batch-size axis
        features = np.concatenate(mol_tensors_list, axis=0)

        return features


class BagofBonds(object):
    """
    The implementation of bag of bonds version of coulomb matrix by katja Hansen et. al. 2015, JPCL.

    Parameters
    ----------
    const : float, optional (default = 1.0)
            The constant value for coordinates unit conversion to atomic unit
            if const=1.0, returns atomic unit
            if const=0.529, returns Angstrom

            const/|Ri-Rj|, which denominator is the euclidean distance
            between two atoms

    n_jobs : int, optional(default=-1)
        The number of parallel processes. If -1, uses all the available processes.

    verbose : bool, optional(default=True)
        The verbosity of messages.

    Attributes
    ----------
    header_ : list of header for the bag of bonds data frame
        contains one nuclear charge (represents single atom) or a tuple of two nuclear charges (represents a bond)

    Examples
    --------
    >>> from chemml.datasets import load_xyz_polarizability
    >>> from chemml.chem import BagofBonds

    >>> coordinates, y = load_xyz_polarizability()
    >>> bob = BagofBonds(const= 1.0)
    >>> features = bob.represent(coordinates)
    """
    def __init__(self, const=1.0, n_jobs = -1, verbose=True):
        self.const = const
        self.n_jobs = n_jobs
        self.verbose = verbose

    def represent(self, molecules):
        """
        provides bag of bonds representation for input molecules.

        Parameters
        ----------
        molecules : chemml.chem.Molecule object or array
            If list, it must be a list of chemml.chem.Molecule objects, otherwise we raise a ValueError.
            In addition, all the molecule objects must provide the XYZ information. Please make sure the XYZ geometry has been
            stored or optimized in advance.

        Returns
        -------
        features : pandas data frame, shape: (n_molecules, max_length_of_combinations)
            The bag of bond features.

        """
        if isinstance(molecules, (list,np.ndarray)):
            molecules = np.array(molecules)
        elif isinstance(molecules, Molecule):
            molecules = np.array([molecules])
        else:
            msg = "The input molecules must be a chemml.chem.Molecule object or a list of objects."
            raise ValueError(msg)

        if molecules.ndim > 1:
            msg = "The molecule must be a chemml.chem.Molecule object or a list of objects."
            raise ValueError(msg)

        # pool of processes
        if self.n_jobs == -1:
            self.n_jobs = cpu_count()
        pool = Pool(processes=self.n_jobs)

        # Create an iterator
        # http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
        def chunks(l, n):
            """Yield successive n-sized chunks from l."""
            for i in range(0, len(l), n):
                yield l[i:i + n]

        # find size of each batch
        batch_size = int(len(molecules) / self.n_jobs)
        if batch_size == 0:
            batch_size = 1

        molecule_chunks = chunks(molecules, batch_size)

        # MAP: CM in parallel
        map_function = partial(self._represent)
        if self.verbose:
            print('featurizing molecules in batches of %i ...' % batch_size)
            pbar = Progbar(len(molecules), width=50)
            bbs_info = []
            for tensors in pool.imap(map_function, molecule_chunks):
                pbar.add(len(tensors[0]))
                bbs_info.append(tensors)
            print('Merging batch features ...    ', end='')
        else:
            bbs_info = pool.map(map_function, molecule_chunks)
        if self.verbose:
            print('[DONE]')

        # REDUCE: Concatenate the obtained tensors
        pool.close()
        pool.join()
        return self.concat_mol_features(bbs_info)

    def _represent(self, molecules):
        BBs_matrix = [] # list of dictionaries for each molecule
        all_keys = {}   # dictionary of unique keys and their maximum length
        for nmol, mol in enumerate(molecules):
            # check molecules
            if not isinstance(mol, Molecule):
                msg = "The input molecules must be chemml.chem.Molecule object with xyz information."
                raise ValueError(msg)

            bags = {}
            mol = np.append(mol.xyz.atomic_numbers, mol.xyz.geometry, axis=1)
            for i in range(len(mol)):
                for j in range(i,len(mol)):
                    if i==j:
                        key = (mol[i,0],)
                        Fc = 0.5 * mol[i, 0] ** 2.4
                        if key in bags:
                            bags[key].append(Fc)
                        else:
                            bags[key] = [Fc]
                    else:
                        key = (max(mol[i,0],mol[j,0]), min(mol[i,0],mol[j,0]))
                        Fc = (mol[i,0]*mol[j,0]*self.const) / np.linalg.norm(mol[i,1:]-mol[j,1:])
                        if key in bags:
                            bags[key].append(Fc)
                        else:
                            bags[key] = [Fc]
            for key in bags:
                if key in all_keys:
                    all_keys[key] = max(all_keys[key], len(bags[key]))
                else:
                    all_keys[key] = len(bags[key])
            BBs_matrix.append(bags)
        return BBs_matrix, all_keys

    def concat_mol_features(self, bbs_info):
        """
        This function concatenates a list of molecules features from parallel run

        Parameters
        ----------
        bbs_info : list or tuple
            The list or tuple of features and keys


        Returns
        -------
        features : data frame
            A single dataframe of all features

        """
        assert isinstance(bbs_info, (tuple, list)), 'Provide a list or tuple of molecule features to concatenate'

        bbs_matrix = []
        all_keys = {}
        for item in bbs_info:
            # join the list of bbs_matrix
            bbs_matrix += item[0]

            # join all the keys
            keys = item[1]
            for key in keys:
                if key in all_keys:
                    all_keys[key] = max(all_keys[key], keys[key])
                else:
                    all_keys[key] = keys[key]

        # pad with zero
        for i, bags in enumerate(bbs_matrix):
            for key in all_keys:
                if key in bags:
                    diff = all_keys[key] - len(bags[key])
                    bbs_matrix[i][key] = sorted(bbs_matrix[i][key], reverse=True) + diff*[0]
                else:
                    bbs_matrix[i][key] = all_keys[key]*[0]

        df = pd.DataFrame(bbs_matrix)
        del bbs_matrix
        order_headers = list(df.columns)
        output = pd.DataFrame(list(df.sum(1)))
        del df

        self.header_ = []
        for key in order_headers:
            self.header_ += all_keys[key] * [key]

        return output
