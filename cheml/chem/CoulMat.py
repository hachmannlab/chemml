import pandas as pd
import numpy as np
import os
import fnmatch
import sys
import scipy


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

class CoulombMatrix(object):
    """ (CoulombMatrix) 
    The implementation of coulomb matrix by Matthias Rupp et al 2012, PRL.
    
    Parameters
    ----------
    CMtype: string, optional (default='SC')
        Available Coulomb Matrix (CM) representations:
            * 'Eigenspectrum' or 'E' 
            * 'Sorted_Coulomb' or 'SC'
            * 'Random_Coulomb' or 'RC'
            * 'BagofBonds' or 'BB'
    
    nPerm: integer, optional (default = 6)
        Number of permutation of coulomb matrix per molecule for Random_Coulomb (RC) 
        type of representation. 

    const: float, optional (default = 1)
            The constant value for coordinates unit conversion to atomic unit
            example: atomic unit -> const=1, Angstrom -> const=0.529
            in action: const/|Ri-Rj|, which denominator is the euclidean distance 
            between atoms i and j coordinates
 
    Returns
    -------
    data set of Coulomb matrix similarities.
    """
    def __init__(self, CMtype='SC', nPerm=6, const=1 ):
        self.CMtype = CMtype
        self.nPerm = nPerm
        self.const = const
        
    def MolfromFile(self, molfile, path=None, reader = 'auto', skip_lines=[2,0], *arguments):
        """ (MolfromFile)
        Construct molecules from one or more input files.
        
        Parameters
        ----------
        molfile: string
            This is the place you define the file name and path. It can contain 
            any special character in the following list:
            
                *       : matches everything
                ?       : matches any single character
                [seq]   : matches any character in seq
                [!seq]  : matches any character not in seq
                /       : filename seperator

            The pattern matching is implemented by fnmatch - Unix filename pattern matching.
            Note: The pattern must include the extension at the end. The only acceptable
            extension is '.xyz'.

        path: string, optional (default = None)            
            If path is None, this function tries to open the file as a single
            file (without any pattern matching). It is a fixed path to subdirectories 
            or files. Therefore, none of the above special characters except '/' 
            can be used.    
            If not None, it determines the path that this function walk through
            and look for every file that matches the pattern in the file. To start
            walking from the curent directory, the path value should be '.'

        reader: string, optional (default = 'auto')
            Available options : 'auto' and 'manual'
            If 'auto', the openbabel readstring function creat the molecule object. 
            The type of files for openbabel class has been set to 'xyz', thus the format
            of the file has to obey the extension as well. However, with 'manual'
            reader you can skip some lines from the beginning or end of xyz files.
            
        skip_lines: list of two integers, optional (default = [2,0])
            Number of lines to skip (int) at the start and end of the xyz files, respectively. 
            Based on the original xyz format only 2 lines from top and zero from
            bottom of a file can be skipped. Thus, number of atoms in the first line 
            can also be ignored. 
            Only available for 'manual' reader. 
            
        *arguments: integer
            If sequences in the special characters should be a range of integers,
            you can easily pass the start and the end arguments of range as extra 
            arguments. But, notice that:
                1- In this case remember to replace seq with %s. Then, you should 
                   have [%s] or [!%s] in the file pattern.
                2- for each %s you must pass two integers (always an even number 
                    of extra arguments must be passed). Also, the order is important
                    and it should be in the same order of '%s's. 
            
        examples
        --------
            (1)
            file: 'Mydir/1f/1_opt.xyz'
            path: None
            >>> sample files to be read: 'Mydir/1f/1_opt.xyz' 
            
            (2)
            file: '[1,2,3,4]?/*_opt.xyz'
            path: 'Mydir'
            >>> sample files to be read: 'Mydir/1f/1_opt.xyz', 'Mydir/2c/2_opt.xyz', ... 
            
            (3)
            file: '[!1,2]?/*_opt.xyz'
            path: '.'
            >>> sample files to be read: './3f/3_opt.xyz', 'Mydir/7c/7_opt.xyz', ... 

            (4)
            file: '*['f','c']/*_opt.xyz'
            path: 'Mydir'
            >>> sample files to be read: 'Mydir/1f/1_opt.xyz', 'Mydir/2c/2_opt.xyz', ... 

            (5)
            file: '[%s]?/[!%s]_opt.xyz'
            path: 'Mydir/all'
            arguments: 1,4,7,10
            >>> sample files to be read: 'Mydir/all/1f/1_opt.xyz', 'Mydir/all/2c/2_opt.xyz', ... 
        """
        file_name, file_extension = os.path.splitext(molfile)
        if file_extension == '':
            msg = 'file extension not determined'
            raise ValueError(msg)
        elif file_extension!='xyz':
            msg = "file extension '%s' not available"%file_extension
            raise ValueError(msg)
        start_end = [arg for arg in arguments]
        if len(start_end) != file_name.count('%s')*2:
            msg = "pass an even number of integers: 2 for each '%s'"
            raise ValueError(msg)        
        
        if path and '%s' in molfile:
            molfile = _add_range(molfile,start_end)
        
        self.molecules = []
        self.max_nAtoms = 1
        if path:
            for root, directories, filenames in os.walk(path):
                file_path = [os.path.join(root, filename) for filename in filenames]
                for fn in fnmatch.filter(file_path, os.path.join(path, molfile)):
                    mol = _file_reader(reader, fn, skip_lines)
                    self.max_nAtoms = max(self.max_nAtoms, len(mol))
                    self.molecules.append(mol)
        else:
            mol = _file_reader(reader, file_name, skip_lines)
            self.max_nAtoms = max(self.max_nAtoms, len(mol))
            self.molecules.append(mol)
            
    def _cal_coul_mat(self, mol):
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
            for m in range(len(mol),self.max_nAtoms):
                vect.append(0.0)
            cm.append(vect)
        for m in range(len(mol),self.max_nAtoms):
            cm.append([0]*self.max_nAtoms)
        return np.array(cm)
    
    def Coulomb_Matrix(self):
        if self.CMtype == 'Eigenspectrum' or self.CMtype == 'E':
            eigenspectrum = []
            for mol in self.molecules:
                cm = self._cal_coul_mat(mol) # Check the constant value for unit conversion; atomic unit -> 1 , Angstrom -> 0.529
                eigenspectrum.append(np.linalg.eigvals(cm).sort()[::-1])
            return np.array(eigenspectrum)
            
        elif self.CMtype == 'Sorted_Coulomb' or self.CMtype == 'SC':
            sorted_cm = []
            for mol in self.molecules:
                cm = self._cal_coul_mat(mol)
                lambdas = [np.linalg.norm(atom) for atom in cm]
                sort_indices = np.argsort(lambdas)
                cm = cm[:,sort_indices][sort_indices,:]
                sorted_cm.append(cm)
            return np.array(sorted_cm)    
                    
        elif self.CMtype == 'Random_Coulomb' or self.CMtype == 'RC':
            random_cm = []
            for nmol,mol in enumerate(self.molecules):
                cm = self._cal_coul_mat(mol)
                lambdas = np.array([np.linalg.norm(atom) for atom in cm])
                sort_indices = np.argsort(lambdas)
                cm = cm[:,sort_indices][sort_indices,:]
                cm_perms = []
                for l in range(self.nPerm):
                    mask = np.random.permutation(self.max_nAtoms)
                    cm_perms.append(cm[:,mask][mask,:])
                random_cm.append(np.array(cm_perms))
            return  np.array(random_cm)
        elif self.CMtype == 'BagofBonds' or self.CMtype == 'BB':
            return self.BagofBonds()

    def BagofBonds(self):
        """ (BagofBonds)
        The implementation of bag of bonds version of coulomb matrix by katja Hansen et al 2015, JPCL.
                
        Returns
        -------
        BBs_matrix : The matrix of bag of bonds for molecules.
        
        """
        BBs_matrix = []
        keys = []
        for m,mol in enumerate(self.molecules):
            bags = [[] for i in range(len(keys))]
            for i in xrange(len(mol)):
                for j in xrange(i,len(mol)):
                    if i==j:
                        key = mol[i,0]*1000+mol[i,0]
                        Fc = 0.5*mol[i,0]**2.4
                    else:
                        key = max(mol[i,0],mol[j,0])*1000 + min(mol[i,0],mol[j,0])
                        Fc = (mol[i,0]*mol[j,0]*self.const) / np.linalg.norm(mol[i,1:]-mol[j,1:])
                    if key not in keys:
                        keys.append(key)
                        bags.append([Fc])
                        for k in xrange(m):
                            BBs_matrix[k].append([])
                    else:
                        bags[keys.index(key)].append(Fc)
            BBs_matrix.append(bags)
        max_len_bags = [max(len(bag) for bag in col) for col in zip(*BBs_matrix)]
        for i,bags in enumerate(all_molecules):
            bagofbags = []
            for j,bag in enumerate(bags):
                bagofbags += sorted(bag, reverse=True) + [0.0]*abs(len(bag)-max_len_bags[j])
            BBs_matrix[i] = bagofbags
        return BBs_matrix            
