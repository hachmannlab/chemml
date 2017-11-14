import pandas as pd
import numpy as np
import sys
# import scipy


def _add_range(file,start_end):
    start = start_end[0::2]
    end = start_end[1::2]
    nit = file.count('%s')
    list_ranges = [str(range(start[it],end[it]+1))[1:-1] for it in range(len(start))]
    s = "pattern = '"+file+"'" + '%(' + str(list_ranges)[1:-1] + ')'
    exec(s)
    return pattern

def _file_reader(reader, filename, skip_lines):        
    if reader == 'auto':    
        sys.path.insert(0,"/user/m27/pkg/openbabel/2.3.2/lib")
        import pybel
        import openbabel
        mol = open(filename, 'r').read()
        mol = pybel.readstring("xyz", mol)
        molecule = [[a.OBAtom.GetAtomicNum(), a.OBAtom.x(), a.OBAtom.y(), a.OBAtom.z()] for a in mol.atoms]
        return np.array(molecule)
    elif reader == 'manual':
        Z = {'Ru': 44.0, 'Re': 75.0, 'Rf': 104.0, 'Rg': 111.0, 'Ra': 88.0, 'Rb': 37.0, 'Rn': 86.0, 'Rh': 45.0, 'Be': 4.0, 'Ba': 56.0, 'Bh': 107.0, 'Bi': 83.0, 'Bk': 97.0, 'Br': 35.0, 'H': 1.0, 'P': 15.0, 'Os': 76.0, 'Es': 99.0, 'Hg': 80.0, 'Ge': 32.0, 'Gd': 64.0, 'Ga': 31.0, 'Pr': 59.0, 'Pt': 78.0, 'Pu': 94.0, 'C': 6.0, 'Pb': 82.0, 'Pa': 91.0, 'Pd': 46.0, 'Cd': 48.0, 'Po': 84.0, 'Pm': 61.0, 'Hs': 108.0, 'Uup': 115.0, 'Uus': 117.0, 'Uuo': 118.0, 'Ho': 67.0, 'Hf': 72.0, 'K': 19.0, 'He': 2.0, 'Md': 101.0, 'Mg': 12.0, 'Mo': 42.0, 'Mn': 25.0, 'O': 8.0, 'Mt': 109.0, 'S': 16.0, 'W': 74.0, 'Zn': 30.0, 'Eu': 63.0, 'Zr': 40.0, 'Er': 68.0, 'Ni': 28.0, 'No': 102.0, 'Na': 11.0, 'Nb': 41.0, 'Nd': 60.0, 'Ne': 10.0, 'Np': 93.0, 'Fr': 87.0, 'Fe': 26.0, 'Fl': 114.0, 'Fm': 100.0, 'B': 5.0, 'F': 9.0, 'Sr': 38.0, 'N': 7.0, 'Kr': 36.0, 'Si': 14.0, 'Sn': 50.0, 'Sm': 62.0, 'V': 23.0, 'Sc': 21.0, 'Sb': 51.0, 'Sg': 106.0, 'Se': 34.0, 'Co': 27.0, 'Cn': 112.0, 'Cm': 96.0, 'Cl': 17.0, 'Ca': 20.0, 'Cf': 98.0, 'Ce': 58.0, 'Xe': 54.0, 'Lu': 71.0, 'Cs': 55.0, 'Cr': 24.0, 'Cu': 29.0, 'La': 57.0, 'Li': 3.0, 'Lv': 116.0, 'Tl': 81.0, 'Tm': 69.0, 'Lr': 103.0, 'Th': 90.0, 'Ti': 22.0, 'Te': 52.0, 'Tb': 65.0, 'Tc': 43.0, 'Ta': 73.0, 'Yb': 70.0, 'Db': 105.0, 'Dy': 66.0, 'Ds': 110.0, 'I': 53.0, 'U': 92.0, 'Y': 39.0, 'Ac': 89.0, 'Ag': 47.0, 'Uut': 113.0, 'Ir': 77.0, 'Am': 95.0, 'Al': 13.0, 'As': 33.0, 'Ar': 18.0, 'Au': 79.0, 'At': 85.0, 'In': 49.0}
        mol = open(filename, 'r').readlines()
        mol = mol[skip_lines[0]:len(mol)-skip_lines[1]]
        molecule = []
        for atom in mol:
            atom = atom.replace('\t',' ')
            atom = atom.strip().split(' ')
            atom = list(filter(lambda x: x!= '', atom))
            molecule.append([Z[atom[0]], float(atom[1]), float(atom[2]), float(atom[3])])
        return np.array(molecule)

class Coulomb_Matrix(object):
    """ (CoulombMatrix) 
    The implementation of coulomb matrix by Matthias Rupp et al 2012, PRL (3 different representations).

    Parameters
    ----------
    CMtype: string, optional (default='SC')
        The coulomb matrix type, one of the following types:
            * 'Eigenspectrum' or 'E' 
            * 'Sorted_Coulomb' or 'SC'
            * 'Random_Coulomb' or 'RC'
    max_n_atoms: integer or 'auto', optional (default = 'auto')
        maximum number of atoms in a molecule. if 'auto' we find it based on the input molecules.

    nPerm: integer, optional (default = 3)
        Number of permutation of coulomb matrix per molecule for Random_Coulomb (RC) 
        type of representation.

    const: float, optional (default = 1)
            The constant value for coordinates unit conversion to atomic unit
            example: atomic unit -> const=1, Angstrom -> const=0.529
            const/|Ri-Rj|, which denominator is the euclidean distance
            between atoms i and j

     """
    def __init__(self, molecules, CMtype='SC', max_n_atoms = 'auto', nPerm=3, const=1):
        self.molecules = molecules
        self.CMtype = CMtype
        self.max_n_atoms = max_n_atoms
        self.nPerm = nPerm
        self.const = const

    def __cal_coul_mat(self, mol):
        cm = []
        for i in xrange(len(mol)):
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

        :param molecules: dictionary or numpy array
            if dictionary: output of cheml.initialization.XYZreader
            if numpy array: array of molecules, which each molecule is an array with shape (number of atoms, 4).
            The four columns are nuclear charge, x-corrdinate, y-coordinate, z-coordinate for each atom, respectively.

        :return: pandas DataFrame
            shape of eigenspectrums (E): (n_molecules, max_n_atoms)
            shape of Sorted_Coulomb (SC): (n_molecules, max_n_atoms*(max_n_atoms+1)/2)
            shape of Random_Coulomb (RC): (n_molecules, nPerm * max_n_atoms * (max_n_atoms+1)/2)

        """
        if isinstance(molecules, dict):
            molecules = np.array([molecules[i]['mol'] for i in molecules])
        elif isinstance(molecules, pd.DataFrame):
            molecules = molecules.values
        if self.max_n_atoms == 'auto':
            self.max_n_atoms = max([len(m) for m in molecules])

        if self.CMtype == 'Eigenspectrum' or self.CMtype == 'E':
            eigenspectrum = []
            for mol in molecules:
                cm = self.__cal_coul_mat(mol) # Check the constant value for unit conversion; atomic unit -> 1 , Angstrom -> 0.529
                eigenspectrum.append(np.linalg.eigvals(cm).sort()[::-1])
            return pd.DataFrame(eigenspectrum)
            
        elif self.CMtype == 'Sorted_Coulomb' or self.CMtype == 'SC':
            sorted_cm = []
            for mol in molecules:
                cm = self.__cal_coul_mat(mol)
                lambdas = [np.linalg.norm(atom) for atom in cm]
                sort_indices = np.argsort(lambdas)
                cm = cm[:,sort_indices][sort_indices,:]
                # sorted_cm.append(cm)
                sorted_cm.append(cm[np.triu_indices(self.max_n_atoms)])
            sorted_cm = pd.DataFrame(sorted_cm)
            return sorted_cm
                    
        elif self.CMtype == 'Random_Coulomb' or self.CMtype == 'RC':
            random_cm = []
            for nmol,mol in enumerate(molecules):
                cm = self.__cal_coul_mat(mol)
                lambdas = np.array([np.linalg.norm(atom) for atom in cm])
                sort_indices = np.argsort(lambdas)
                cm = cm[:,sort_indices][sort_indices,:]
                cm_perms = []
                for l in range(self.nPerm):
                    mask = np.random.permutation(self.max_n_atoms)
                    cm_mask = cm[:,mask][mask,:]
                    # cm_perms.append(cm_mask)
                    cm_perms += list(cm_mask[np.triu_indices(self.max_n_atoms)])
                random_cm.append(np.array(cm_perms))
            return pd.DataFrame(random_cm)

class Bag_of_Bonds(object):
    """ (Bag_of_Bonds)
    The implementation of bag of bonds version of coulomb matrix by katja Hansen et al 2015, JPCL.

    Parameters
    ----------
    const: float, optional (default = 1)
            The constant value for coordinates unit conversion to atomic unit
            if const=1, returns atomic unit
            if const=0.529, returns Angstrom

            const/|Ri-Rj|, which denominator is the euclidean distance
            between atoms i and j

    Attributes
    ----------
    headers: list of headers for the bag of bonds data frame
        contains one nuclear charge (represents single atom) or a tuple of two nuclear charges (represents a bond)
    """
    def __init__(self, const=1):
        self.const = const

    def represent(self, molecules):
        """
        provides bag og bonds representation for input molecules.

        :param molecules: dictionary or numpy array
            if dictionary: output of cheml.initialization.XYZreader
            if numpy array: array of molecules, which each molecule is an array with shape (number of atoms, 4).
            The four columns are nuclear charge, x-corrdinate, y-coordinate, z-coordinate for each atom, respectively.

        :return: pandas data frame
            shape of bag of bonds: (n_molecules, cumulative_length_of_combinations)

        """
        if isinstance(molecules, dict):
            molecules = np.array([molecules[i]['mol'] for i in molecules])
        elif isinstance(molecules, pd.DataFrame):
            molecules = molecules.values


        BBs_matrix = [] # list of dictionaries for each molecule
        all_keys = {}   # dictionary of unique keys and their maximum length
        for nmol,mol in enumerate(molecules):
            bags = {}
            for i in xrange(len(mol)):
                for j in xrange(i,len(mol)):
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
        self.headers = []
        for key in order_headers:
            if key[0]==key[1]:
                k = key[0]
            else:
                k = key
            self.headers += all_keys[key] * [k]
        return output