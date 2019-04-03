from builtins import range
import pandas as pd
import numpy as np

from chemml.chem import Molecule

class CoulombMatrix(object):
    """
    The implementation of coulomb matrix descriptors by Matthias Rupp et. al. 2012, PRL (All 3 different variations).

    Parameters
    ----------
    CMtype: str, optional (default='SC')
        The coulomb matrix type, one of the following types:
            * 'Unsorted_Matrix' or 'UM'
            * 'Unsorted_Triangular' or 'UT'
            * 'Eigenspectrum' or 'E' 
            * 'Sorted_Coulomb' or 'SC'
            * 'Random_Coulomb' or 'RC'
    max_n_atoms: int or 'auto', optional (default = 'auto')
        Set the maximum number of atoms per molecule (to which all representations will be padded).
        If 'auto', we find it based on all input molecules.
    nPerm: int, optional (default = 3)
        Number of permutation of coulomb matrix per molecule for Random_Coulomb (RC) 
        type of representation.
    const: float, optional (default = 1)
            The constant value for coordinates unit conversion to atomic unit
            example: atomic unit -> const=1, Angstrom -> const=0.529
            const/|Ri-Rj|, which denominator is the euclidean distance
            between atoms i and j
    """
    def __init__(self, CMtype='SC', max_n_atoms = 'auto', nPerm=3, const=1):
        self.CMtype = CMtype
        self.max_n_atoms = max_n_atoms
        self.nPerm = nPerm
        self.const = const

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
            for m in range(len(mol),self.max_n_atoms):
                vect.append(0.0)
            cm.append(vect)
        for m in range(len(mol),self.max_n_atoms):
            cm.append([0]*self.max_n_atoms)
        return np.array(cm) #shape nAtoms*nAtoms

    def represent(self, molecules):
        """
        provides coulomb matrix representation for input molecules.

        Parameters
        ----------
        molecules: chemml.chem.Molecule object or list
            If list, it must be a list of chemml.chem.Molecule objects, otherwise we raise a ValueError.
            In addition, all the molecule objects must provide the XYZ information. Please make sure the XYZ geometry has been
            stored or optimized in advance.

        Returns
        -------
        Pandas DataFrame
            A data frame with same number of rows as number of molecules will be returned.
            The exact shape of the dataframe depends on the type of CM as follows:
                - shape of Unsorted_Matrix (UM): (n_molecules, max_n_atoms**2)
                - shape of Unsorted_Triangular (UT): (n_molecules, max_n_atoms*(max_n_atoms+1)/2)
                - shape of eigenspectrums (E): (n_molecules, max_n_atoms)
                - shape of Sorted_Coulomb (SC): (n_molecules, max_n_atoms*(max_n_atoms+1)/2)
                - shape of Random_Coulomb (RC): (n_molecules, nPerm * max_n_atoms * (max_n_atoms+1)/2)
        """
        if isinstance(molecules, list):
            molecules = np.array(molecules)
        elif isinstance(molecules, Molecule):
            molecules = np.array([molecules])
        else:
            msg = "The molecule must be a chemml.chem.Molecule object or a list of objets."
            raise ValueError(msg)

        if molecules.ndim >1:
            msg = "The molecule must be a chemml.chem.Molecule object or a list of objets."
            raise ValueError(msg)

        self.n_molecules = molecules.shape[0]

        # max number of atoms based on the list of molecules
        if self.max_n_atoms == 'auto':
            try:
                self.max_n_atoms = max([m.xyz.atomic_numbers.shape[0] for m in molecules])
            except:
                msg = "The molecule must be a chemml.chem.Molecule object or a list of objets."
                raise ValueError(msg)

        if self.CMtype == "Unsorted_Matrix" or self.CMtype == 'UM':
            cms = np.array([])
            for mol in molecules:
                cm = self.__cal_coul_mat(mol)
                cms = np.append(cms, cm.ravel())
            cms = cms.reshape(self.n_molecules, self.max_n_atoms**2)
            cms = pd.DataFrame(cms)
            return cms

        elif self.CMtype == "Unsorted_Triangular" or self.CMtype == 'UT':
            cms = np.array([])
            for mol in molecules:
                cm = self.__cal_coul_mat(mol)
                cms = np.append(cms, cm[np.tril_indices(self.max_n_atoms)])
            cms = cms.reshape(self.n_molecules, int(self.max_n_atoms*(self.max_n_atoms+1)/2))
            cms = pd.DataFrame(cms)
            return cms

        elif self.CMtype == 'Eigenspectrum' or self.CMtype == 'E':
            eigenspectrums = np.array([])
            for mol in molecules:
                cm = self.__cal_coul_mat(mol) # Check the constant value for unit conversion; atomic unit -> 1 , Angstrom -> 0.529
                eig = np.linalg.eigvals(cm)
                eig[::-1].sort()
                eigenspectrums = np.append(eigenspectrums,eig)
            eigenspectrums = eigenspectrums.reshape(self.n_molecules, self.max_n_atoms)
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
                sorted_cm = np.append(sorted_cm, cm[np.tril_indices(self.max_n_atoms)]) # lower-triangular
            sorted_cm = sorted_cm.reshape(self.n_molecules, int(self.max_n_atoms*(self.max_n_atoms+1)/2))
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
                    mask = np.random.permutation(self.max_n_atoms)
                    cm_mask = cm[:,mask][mask,:]
                    # cm_perms.append(cm_mask)
                    cm_perms += list(cm_mask[np.tril_indices(self.max_n_atoms)]) # lower-triangular
                random_cm = np.append(random_cm, cm_perms)
            random_cm = random_cm.reshape(self.n_molecules, int(self.nPerm*self.max_n_atoms*(self.max_n_atoms+1)/2))
            return pd.DataFrame(random_cm)

class BagofBonds(object):
    """
    The implementation of bag of bonds version of coulomb matrix by katja Hansen et. al. 2015, JPCL.

    Parameters
    ----------
    const: float, optional (default = 1.0)
            The constant value for coordinates unit conversion to atomic unit
            if const=1.0, returns atomic unit
            if const=0.529, returns Angstrom

            const/|Ri-Rj|, which denominator is the euclidean distance
            between two atoms

    Attributes
    ----------
    header_: list of header for the bag of bonds data frame
        contains one nuclear charge (represents single atom) or a tuple of two nuclear charges (represents a bond)

    Examples
    --------
    >>> from chemml.datasets import load_xyz_polarizability
    >>> from chemml.chem import BagofBonds

    >>> coordinates, y = load_xyz_polarizability()
    >>> bob = BagofBonds(const= 1.0)
    >>> X = bob.represent(coordinates)
    """
    def __init__(self, const=1.0):
        self.const = const

    def represent(self, molecules):
        """
        provides bag of bonds representation for input molecules.

        Parameters
        ----------
        molecules: chemml.chem.Molecule object or list
            If list, it must be a list of chemml.chem.Molecule objects, otherwise we raise a ValueError.
            In addition, all the molecule objects must provide the XYZ information. Please make sure the XYZ geometry has been
            stored or optimized in advance.

        Returns
        -------
        pandas data frame, shape: (n_molecules, max_length_of_combinations)

        """
        if isinstance(molecules, list):
            molecules = np.array(molecules)
        elif isinstance(molecules, Molecule):
            molecules = np.array([molecules])
        else:
            msg = "The molecule must be a chemml.chem.Molecule object or a list of objets."
            raise ValueError(msg)

        if molecules.ndim > 1:
            msg = "The molecule must be a chemml.chem.Molecule object or a list of objets."
            raise ValueError(msg)

        BBs_matrix = [] # list of dictionaries for each molecule
        all_keys = {}   # dictionary of unique keys and their maximum length
        for nmol,mol in enumerate(molecules):
            bags = {}
            mol = np.append(mol.xyz.atomic_numbers, mol.xyz.geometry, axis=1)
            for i in range(len(mol)):
                for j in range(i,len(mol)):
                    if i==j:
                        key = (mol[i,0], mol[i,0])
                        Fc = 0.5 * mol[i, 0] ** 2.4
                        if key in bags:
                            bags[key].append(Fc)
                        else:
                            bags[key] = [Fc]
                    else:
                        key = (max(mol[i,0],mol[j,0]),min(mol[i,0],mol[j,0]))
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

        # extend bags with zero
        for i, bags in enumerate(BBs_matrix):
            for key in all_keys:
                if key in bags:
                    diff = all_keys[key] - len(bags[key])
                    BBs_matrix[i][key] = sorted(BBs_matrix[i][key], reverse=True) + diff*[0]
                else:
                    BBs_matrix[i][key] = all_keys[key]*[0]
        df = pd.DataFrame(BBs_matrix)
        del BBs_matrix
        order_headers = list(df.columns)
        output = pd.DataFrame(list(df.sum(1)))
        del df
        self.header_ = []
        for key in order_headers:
            if key[0]==key[1]:
                k = key[0]
            else:
                k = key
            self.header_ += all_keys[key] * [k]
        return output
