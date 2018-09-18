import pandas as pd
import numpy as np
import os
import fnmatch

# TODO: AP/bit is time consuming - should be used directly with distance matrix

def _add_range(file,start_end):
    start = start_end[0::2]
    end = start_end[1::2]
    nit = file.count('%s')
    list_ranges = [str(range(start[it],end[it]+1))[1:-1] for it in range(len(start))]
    s = "pattern = '"+file+"'" + '%(' + str(list_ranges)[1:-1] + ')'
    exec(s)
    return pattern

class RDKitFingerprint(object):
    """ An interface to RDKit fingerprints.
    
    Parameters
    ----------
    FPtype: string, optional (default='Morgan')
        Available fingerprint types:
            - 'Hashed_atom_pair' or 'HAP' 
            - 'Atom_pair' or 'AP'
            - 'MACCS'
            - 'Morgan'
            - 'Hashed_topological_torsion' or 'HTT'
            - 'Topological_torsion' or 'TT' 

    vector: string, optional (default = 'bit')
        Available options for vector:
            - 'int' : including counts for each bit instead of just zeros and ones
                    It is not available for 'MACCS'.
            - 'bit' : only zeros and ones
                    It is not available for 'Topological_torsion'.

    nBits: integer, optional (default = 1024)
        It sets number of bits in the specific type of fingerprint vectors. 
        Not availble for: 
            - 'MACCS' (MACCS keys have a fixed length of 167 bits)
            - 'Atom_pair' (In the Atom_pair we return only those bits that are 
                           non zero for at least one of the molecules. Thus, the size 
                           will be significantly less compare to more than 8 million  
                           available bits)
            - 'Topological_torsion'  

    radius: integer, optional (default = 2)
        only availble for 'Morgan' fingerprint.

    removeHs: boolean, optional (default=True)
        If True, remove any hydrogen from the graph of a molecule.

    Returns
    -------
    data set of fingerprint vectors.
    """
    def __init__(self, removeHs=True, FPtype='Morgan', vector='bit', nBits=1024, 
                 radius = 2):
        self.FPtype = FPtype
        self.vector = vector
        self.nBits = nBits
        self.radius = radius
        self.removeHs = removeHs

    def MolfromFile(self, molfile, path=None, *arguments):
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
            
        examples
        --------
            (1)
            file: 'Mydir/1f/1_opt.smi'
            path: None
            >>> sample files to be read: 'Mydir/1f/1_opt.smi' 
            
            (2)
            file: '[1,2,3,4]?/*_opt.mol'
            path: 'Mydir'
            >>> sample files to be read: 'Mydir/1f/1_opt.mol', 'Mydir/2c/2_opt.mol', ... 
            
            (3)
            file: '[!1,2]?/*_opt.pdb'
            path: '.'
            >>> sample files to be read: './3f/3_opt.pdb', 'Mydir/7c/7_opt.pdb', ... 

            (4)
            file: '*['f','c']/*_opt.tpl'
            path: 'Mydir'
            >>> sample files to be read: 'Mydir/1f/1_opt.tpl', 'Mydir/2c/2_opt.tpl', ... 

            (5)
            file: '[%s]?/[!%s]_opt.mol2'
            arguments: 1,4,7,10
            path: 'Mydir/all'
            >>> sample files to be read: 'Mydir/all/1f/1_opt.mol2', 'Mydir/all/2c/2_opt.mol2', ... 
        """
        from rdkit import Chem
        extensions = {'.mol':       Chem.MolFromMolFile,
                      '.mol2':      Chem.MolFromMol2File,
                      '.pdb':       Chem.MolFromPDBFile,
                      '.tpl':       Chem.MolFromTPLFile,
                      '.smi':       Chem.MolFromSmiles,
                      '.smarts':    Chem.MolFromSmarts,
                      # '.inchi':     Chem.MolFromInchi
                      }
        file_name, file_extension = os.path.splitext(molfile)
        if file_extension == '':
            msg = 'file extension not determined'
            raise ValueError(msg)
        elif file_extension not in extensions:
            msg = "file extension '%s' not available"%file_extension
            raise ValueError(msg)
        start_end = [arg for arg in arguments]
        if len(start_end) != file_name.count('%s')*2:
            msg = "pass an even number of integers: 2 for each '%s'"
            raise ValueError(msg)        
        
        if path and '%s' in molfile:
            molfile = _add_range(molfile,start_end)

        self.removed_rows = []
        self.molecules = []
        if path:
            for root, directories, filenames in os.walk(path):
                file_path = [os.path.join(root, filename) for filename in filenames]
                for filename in fnmatch.filter(file_path, os.path.join(path, molfile)):
                    if file_extension in ['.smi','.smarts']:
                        mols = open(filename, 'r')
                        mols = mols.readlines()
                        for i, x in enumerate(mols):
                            mol = extensions[file_extension](x.strip(),removeHs=self.removeHs)
                            if mol is None:
                                self.removed_rows.append(i)
                            else:
                                self.molecules.append(mol)
                        # mols = [extensions[file_extension](x.strip(),removeHs=self.removeHs) for x in mols]
                        # self.molecules += mols
                    else:
                        self.molecules += [extensions[file_extension](filename,removeHs=self.removeHs)]
        else:
            if file_extension in ['.smi','.smarts']:
                mols = open(molfile, 'r')
                mols = mols.readlines()
                for i,x in enumerate(mols):
                    mol = extensions[file_extension](x.strip())
                    if mol is None:
                        self.removed_rows.append(i)
                    else:
                       self.molecules.append(mol)
                # mols = [extensions[file_extension](x.strip()) for x in mols]
                # self.molecules += mols
            else:
                self.molecules += [extensions[file_extension](filename,removeHs=self.removeHs)]

    def Fingerprint(self):
        if self.FPtype == 'Hashed_atom_pair' or self.FPtype == 'HAP':
            if self.vector == 'int':
                from rdkit.Chem.AtomPairs.Pairs import GetHashedAtomPairFingerprint
                self.fps = [GetHashedAtomPairFingerprint(m,nBits=self.nBits) for m in self.molecules]
                dict_nonzero = [fp.GetNonzeroElements() for fp in self.fps]
                data = pd.DataFrame(dict_nonzero,columns=range(self.nBits))
                data.fillna(0, inplace = True)
                return data
            elif self.vector == 'bit':
                from rdkit.Chem.rdMolDescriptors import GetHashedAtomPairFingerprintAsBitVect
                self.fps = [GetHashedAtomPairFingerprintAsBitVect(m,nBits=self.nBits) for m in self.molecules]
                data = np.array(self.fps)
                data = pd.DataFrame(data)
                return data
            else:
                msg = "The argument vector can be 'int' or 'bit'"
                raise ValueError(msg)
        elif self.FPtype == 'Atom_pair' or self.FPtype == 'AP':
            if self.vector == 'int':
                from rdkit.Chem.AtomPairs.Pairs import GetAtomPairFingerprint
                self.fps = [GetAtomPairFingerprint(m) for m in self.molecules]
                dict_nonzero = [fp.GetNonzeroElements() for fp in self.fps]
                pairScores = []
                for fp in dict_nonzero:
                    pairScores += [key for key in fp]
                data = pd.DataFrame(dict_nonzero,columns=list(set(pairScores)))
                data.fillna(0, inplace = True)
                return data
            elif self.vector == 'bit':
                from rdkit.Chem.AtomPairs.Pairs import GetAtomPairFingerprintAsBitVect
                self.fps = [GetAtomPairFingerprintAsBitVect(m) for m in self.molecules]
                data = np.array(self.fps)
                data = pd.DataFrame(data)

                print len(data.columns)
                d_des = data.describe()
                for i in data.columns:
                    if d_des[i]['mean'] == 0 :
                        data.drop(i,1)
                print len(data.columns)

                dict_nonzero = []
                for fp in self.fps:
                    dict_nonzero.append({i:el for i,el in enumerate(fp) if el!=0})
                pairScores = []
                for fp in dict_nonzero:
                    pairScores += [key for key in fp]
                data = pd.DataFrame(dict_nonzero,columns=list(set(pairScores)))
                data.fillna(0, inplace = True)
                return data
            else:
                msg = "The argument vector can be 'int' or 'bit'"
                raise ValueError(msg)
        elif self.FPtype == 'MACCS':
            if self.vector == 'int':
                msg = "There is no RDKit function to encode int vectors for MACCS keys"
                raise ValueError(msg)
            elif self.vector == 'bit':
                from rdkit.Chem.MACCSkeys import GenMACCSKeys
                self.fps = [GenMACCSKeys(mol) for mol in self.molecules]
                data = np.array(self.fps)
                data = pd.DataFrame(data)
                return data
            else:
                msg = "The vector argument can only be 'int' or 'bit'"
                raise ValueError(msg)
        elif self.FPtype == 'Morgan':
            if self.vector == 'int':
                from rdkit.Chem.rdMolDescriptors import GetMorganFingerprint
                self.fps = [GetMorganFingerprint(mol, self.radius) for mol in self.molecules]
                dict_nonzero = [fp.GetNonzeroElements() for fp in self.fps]
                pairScores = []
                for fp in dict_nonzero:
                    pairScores += list(fp)
                data = pd.DataFrame(dict_nonzero,columns=list(set(pairScores)))
                data.fillna(0, inplace = True)
                return data
            elif self.vector == 'bit':
                from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
                self.fps = [GetMorganFingerprintAsBitVect(mol, self.radius,nBits=self.nBits) for mol in self.molecules]
                data = np.array(self.fps)
                data = pd.DataFrame(data)
                return data
            else:
                msg = "The argument vector can only be 'int' or 'bit'"
                raise ValueError(msg)
        elif self.FPtype == 'Hashed_topological_torsion' or self.FPtype == 'HTT':
            if self.vector == 'int':
                from rdkit.Chem.rdMolDescriptors import GetHashedTopologicalTorsionFingerprint
                self.fps = [GetHashedTopologicalTorsionFingerprint(m,nBits=self.nBits) for m in self.molecules]
                dict_nonzero = [fp.GetNonzeroElements() for fp in self.fps]
                data = pd.DataFrame(dict_nonzero,columns=range(self.nBits))
                data.fillna(0, inplace = True)
                return data
            elif self.vector == 'bit':
                from rdkit.Chem.rdMolDescriptors import GetHashedTopologicalTorsionFingerprintAsBitVect
                self.fps = [GetHashedTopologicalTorsionFingerprintAsBitVect(m,nBits=self.nBits) for m in self.molecules]
                data = np.array(self.fps)
                data = pd.DataFrame(data)
                return data
            else:
                msg = "The argument vector can be 'int' or 'bit'"
                raise ValueError(msg)
        elif self.FPtype == 'Topological_torsion' or self.FPtype == 'TT':
            if self.vector == 'int':
                from rdkit.Chem.AtomPairs.Torsions import GetTopologicalTorsionFingerprintAsIntVect
                self.fps = [GetTopologicalTorsionFingerprintAsIntVect(mol) for mol in self.molecules]
                dict_nonzero = [fp.GetNonzeroElements() for fp in self.fps]
                pairScores = []
                for fp in dict_nonzero:
                    pairScores += list(fp)
                data = pd.DataFrame(dict_nonzero,columns=list(set(pairScores)))
                data.fillna(0, inplace = True)
                return data
            elif self.vector == 'bit':
                msg = "There is no RDKit function to encode bit vectors for Topological Torsion Fingerprints"
                raise ValueError(msg)
            else:
                msg = "The argument vector can only be 'int'"
                raise ValueError(msg)
        else:
            msg = "The type argument '%s' is not a valid fingerprint type"%self.FPtype
            raise ValueError(msg)
    
    # def Similarity(self,data=None, metric='Euclidean', nCores = 1):
    #     """(Similarity)
    #     Calculate molecular similarity matrix, while each row of data is a feature
    #     vector of one molecule.
    #
    #     Parameters
    #     ----------
    #     data: pandas dataframe, optional (default = None)
    #         Each row of data represents corresponding features of one molecule (sample).
    #
    #     metric: string, optional (default = 'Euclidean')
    #         Similarity metric. Available similarity metrics include Euclidean,
    #         Tanimoto, Dice, Cosine, Sokal, Russel, Kulczynski, McConnaughey, and Tversky.
    #
    #         Note:
    #             if data = None, all metrics except 'Euclidean' can be used. The
    #             Fingerprint function must be called first. Thus, the similarity metrics
    #             implemetned by RDKit will be used on calculated fingerprints.
    #
    #             if a dataframe has been passed for data, the only available metric
    #             is 'Euclidean'.
    #
    #     nCores: integer, optional (default = 1)
    #         Number of cores for multiprocessing.
    #     """
    #     if data == None:
    #         if metric in ['Tanimoto','Dice','Cosine','Sokal','Russel','Kulczynski','McConnaughey','Tversky']:
    #             from rdkit import DataStructs
    #
    #         else:
    #             msg = "The metric '%s' is not a valid metric type for "%metric
    #             raise ValueError(msg)
    #     return None