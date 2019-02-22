from __future__ import print_function
from rdkit import Chem
from rdkit.Chem import AllChem
import warnings
import numpy as np

from ..utils import update_default_kwargs


class XYZ(object):
    """
    This class stores the information that is typically carried by standard XYZ files.

    Parameters
    ----------
    geometry: ndarray
        The numpy array of shape (number_of_atoms, 3).
        It stores the xyz coordinates for each atom of the molecule.

    atomic_numbers: ndarray
        The numpy array of shape (number_of_atoms, 1).
        It stores the atomic numbers of each atom in the molecule (in the same order as geometry).

    atomic_symbols: ndarray
        The numpy array of shape (number_of_atoms, 1).
        It stores the atomic symbols of each atom in the molecule (in the same order as geometry).

    """
    def __init__(self,geometry, atomic_numbers, atomic_symbols):
        # check data type
        if not (isinstance(geometry, np.ndarray) and isinstance(atomic_numbers,np.ndarray)
                and isinstance(atomic_symbols, np.ndarray)):
            msg = "The parameters' value must be numpy array."
            raise ValueError(msg)
        else:
            # check dimensions
            if geometry.ndim != 2 or atomic_symbols.ndim != 2 or atomic_numbers.ndim != 2:
                msg = "The parameters' arrays must be 2 dimensional."
                raise ValueError(msg)
        # cast data format
        try:
            geometry.astype(float)
            atomic_numbers.astype(int)
            atomic_symbols.astype(str)
        except ValueError:
            msg = "The input data must be able to be casted to the following specified types:\n" \
                  "    - geoometry      >> float\n" \
                  "    - atomic_numbers >> integer\n" \
                  "    - atomic_symbols >> string"
            raise ValueError(msg)
        # check shapes
        if geometry.shape[1] == 3:
            if geometry.shape[0] == atomic_symbols.shape[0] and \
               geometry.shape[0] == atomic_numbers.shape[0] :
                self.geometry = geometry
                self.atomic_numbers = atomic_numbers
                self.atomic_symbols = atomic_symbols
            else:
                msg = "The number of atoms in the molecule is different between entries."
                raise ValueError(msg)
        else:
            msg = "All three coordinate positions are required in the geometries."
            raise ValueError(msg)


class Molecule(object):
    """
    The Molecule class powered by RDKit Molecule functionalities.

    Attributes
    ----------
    rdkit_molecule: object
        The `rdkit.Chem.rdchem.Mol` object

    smiles: str
        The SMILES string that you get by running the `to_smiles` method.

    smarts: str
        The SMARTS string that you get by running the `to_smarts` method.

    inchi: str
        The InChi string that you get by running the `to_inchi` method.

    xyz: instance of <class 'chemml.chem.molecule.XYZ'>
        The class that stores the xyz-type info. The available arguments in the class are geometry,
        atomic_numbers, and atomic_symbols.

    Examples
    --------
    >>> from chemml.chem import Molecule
    >>> caffeine_smiles = 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'
    >>> caffeine_smarts = '[#6]-[#7]1:[#6]:[#7]:[#6]2:[#6]:1:[#6](=[#8]):[#7](:[#6](=[#8]):[#7]:2-[#6])-[#6]'
    >>> caffeine_inchi = 'InChI=1S/C8H10N4O2/c1-10-4-9-6-5(10)7(13)12(3)8(14)11(6)2/h4H,1-3H3'
    >>> mol = Molecule()
    >>> mol.load(caffeine_smiles, 'smiles')
    >>> mol
    <Molecule(rdkit_molecule : <rdkit.Chem.rdchem.Mol object at 0x11060a8a0>,
          creator        : ('SMILES', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'),
          smiles         : 'Cn1c(=O)c2c(ncn2C)n(C)c1=O',
          smarts         : None,
          inchi          : None,
          xyz            : None)>
    >>> mol.smiles # this is the canonical SMILES by default
    'Cn1c(=O)c2c(ncn2C)n(C)c1=O'
    >>> mol.creator
    ('SMILES', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C')
    >>> mol.to_smiles(kekuleSmiles=True)
    >>> mol
    <Molecule(rdkit_molecule : <rdkit.Chem.rdchem.Mol object at 0x11060a8a0>,
          creator        : ('SMILES', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'),
          smiles         : 'CN1C(=O)C2=C(N=CN2C)N(C)C1=O',
          smarts         : None,
          inchi          : None,
          xyz            : None)>
    >>> mol.smiles # the kukule SMILES is not canonical
    'CN1C(=O)C2=C(N=CN2C)N(C)C1=O'
    >>> mol.inchi is None
    True
    >>> mol.to_inchi()
    >>> mol
    <Molecule(rdkit_molecule : <rdkit.Chem.rdchem.Mol object at 0x11060a8a0>,
          creator        : ('SMILES', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'),
          smiles         : 'CN1C(=O)C2=C(N=CN2C)N(C)C1=O',
          smarts         : None,
          inchi          : 'InChI=1S/C8H10N4O2/c1-10-4-9-6-5(10)7(13)12(3)8(14)11(6)2/h4H,1-3H3',
          xyz            : None)>
    >>> mol.inchi
    'InChI=1S/C8H10N4O2/c1-10-4-9-6-5(10)7(13)12(3)8(14)11(6)2/h4H,1-3H3'
    >>> mol.to_smarts(isomericSmiles=False)
    >>> mol.smarts
    '[#6]-[#7]1:[#6]:[#7]:[#6]2:[#6]:1:[#6](=[#8]):[#7](:[#6](=[#8]):[#7]:2-[#6])-[#6]'
    >>>
    >>> # add hydrogens and recreate smiles and inchi
    >>> mol.hydrogens('add')
    >>> mol.to_smiles()
    >>> mol.to_inchi()
    <Molecule(rdkit_molecule : <rdkit.Chem.rdchem.Mol object at 0x11060ab70>,
          creator        : ('SMILES', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'),
          smiles         : '[H]c1nc2c(c(=O)n(C([H])([H])[H])c(=O)n2C([H])([H])[H])n1C([H])([H])[H]',
          smarts         : '[#6]-[#7]1:[#6]:[#7]:[#6]2:[#6]:1:[#6](=[#8]):[#7](:[#6](=[#8]):[#7]:2-[#6])-[#6]',
          inchi          : 'InChI=1S/C8H10N4O2/c1-10-4-9-6-5(10)7(13)12(3)8(14)11(6)2/h4H,1-3H3',
          xyz            : None)>
    >>> # note that by addition of hydrogens, the smiles string changed, but not inchi
    >>> mol.to_xyz()
    ValueError: The conformation has not been built yet. Maybe due to the 2D representation of the creator.
    You should set the optimizer value if you wish to embed and optimize the 3D geometries.
    >>> mol.to_xyz('MMFF', maxIters=300, mmffVariant='MMFF94s')
    >>> mol
    <Molecule(rdkit_molecule : <rdkit.Chem.rdchem.Mol object at 0x11060ab70>,
          creator        : ('SMILES', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'),
          smiles         : '[H]c1nc2c(c(=O)n(C([H])([H])[H])c(=O)n2C([H])([H])[H])n1C([H])([H])[H]',
          smarts         : '[#6]-[#7]1:[#6]:[#7]:[#6]2:[#6]:1:[#6](=[#8]):[#7](:[#6](=[#8]):[#7]:2-[#6])-[#6]',
          inchi          : 'InChI=1S/C8H10N4O2/c1-10-4-9-6-5(10)7(13)12(3)8(14)11(6)2/h4H,1-3H3',
          xyz            : <chemml.chem.molecule.XYZ object at 0x1105f34e0>)>
    >>> mol.xyz
    <chemml.chem.molecule.XYZ object at 0x1105f34e0>
    >>> mol.xyz.geometry
    array([[-3.13498321,  1.08078307,  0.33372515],
       [-2.15703638,  0.0494926 ,  0.0969075 ],
       [-2.41850776, -1.27323453, -0.14538583],
       [-1.30933543, -1.96359887, -0.32234393],
       [-0.31298208, -1.04463051, -0.1870366 ],
       [-0.80033326,  0.20055608,  0.07079013],
       [ 0.02979071,  1.33923464,  0.25484381],
       [-0.41969338,  2.45755985,  0.48688003],
       [ 1.39083332,  1.04039479,  0.14253256],
       [ 1.93405927, -0.23430839, -0.12320318],
       [ 3.15509616, -0.39892779, -0.20417755],
       [ 1.03516373, -1.28860035, -0.28833489],
       [ 1.51247526, -2.63123373, -0.56438513],
       [ 2.34337825,  2.12037198,  0.31039958],
       [-3.03038469,  1.84426033, -0.44113804],
       [-2.95880047,  1.50667225,  1.32459816],
       [-4.13807215,  0.64857048,  0.29175711],
       [-3.4224011 , -1.6776074 , -0.18199971],
       [ 2.60349515, -2.67375289, -0.61674561],
       [ 1.10339164, -2.96582894, -1.52299068],
       [ 1.17455601, -3.30144289,  0.23239402],
       [ 2.94406381,  2.20916763, -0.60086251],
       [ 1.86156872,  3.07985237,  0.51337935],
       [ 3.01465788,  1.87625024,  1.14039627]])
    >>> mol.xyz.atomic_numbers
    array([[6],
       [7],
       [6],
       [7],
       [6],
       [6],
       [6],
       [8],
       [7],
       [6],
       [8],
       [7],
       [6],
       [6],
       [1],
       [1],
       [1],
       [1],
       [1],
       [1],
       [1],
       [1],
       [1],
       [1]])
    >>> mol.xyz.atomic_symbols
    array([['C'],
           ['N'],
           ['C'],
           ['N'],
           ['C'],
           ['C'],
           ['C'],
           ['O'],
           ['N'],
           ['C'],
           ['O'],
           ['N'],
           ['C'],
           ['C'],
           ['H'],
           ['H'],
           ['H'],
           ['H'],
           ['H'],
           ['H'],
           ['H'],
           ['H'],
           ['H'],
           ['H']], dtype='<U1')
    """
    def __init__(self):
        self.rdkit_molecule = None
        self.creator = None
        self._init_attributes()
        self._extra_docs()

    def __repr__(self):
        return '<Molecule(rdkit_molecule : {self.rdkit_molecule!r},\n' \
               '          creator        : {self.creator!r},\n' \
               '          smiles         : {self.smiles!r},\n' \
               '          smarts         : {self.smarts!r},\n' \
               '          inchi          : {self.inchi!r},\n' \
               '          xyz            : {self.xyz!r})>'.format(self=self)

    def _init_attributes(self):
        # SMILES
        self._smiles = None
        self._smiles_args = None
        # SMARTS
        self._smarts = None
        self._smarts_args = None
        # Inchi
        self._inchi = None
        self._inchi_args = None
        # xyz
        self._xyz = None
        self._UFF_args = None
        self._MMFF_args = None

    def _extra_docs(self):
        # method: to_smiles
        self._to_smiles_core_names = ("rdkit.Chem.MolToSmarts",)
        self._to_smiles_core_docs = ("http://rdkit.org/docs/source/rdkit.Chem.inchi.html?highlight=inchi#rdkit.Chem.inchi.MolToInchi",)
        # method: to_smarts
        self._to_smarts_core_names = ("rdkit.Chem.MolToSmarts",)
        self._to_smarts_core_docs = ("http://rdkit.org/docs/source/rdkit.Chem.inchi.html?highlight=inchi#rdkit.Chem.inchi.MolToInchi",)
        # method: to_inchi
        self._to_inchi_core_names = ("rdkit.Chem.MolToInchi",)
        self._to_inchi_core_docs = ("http://rdkit.org/docs/source/rdkit.Chem.inchi.html?highlight=inchi#rdkit.Chem.inchi.MolToInchi",)
        # method: hydrogens
        self._hydrogens_core_names = ("rdkit.Chem.AddHs","rdkit.Chem.RemoveHs")
        self._hydrogens_core_docs = ("http://rdkit.org/docs/source/rdkit.Chem.rdmolops.html?highlight=addhs#rdkit.Chem.rdmolops.AddHs",
                                    "http://rdkit.org/docs/source/rdkit.Chem.rdmolops.html?highlight=addhs#rdkit.Chem.rdmolops.RemoveHs")
        #
        self._to_xyz_core_names = ("rdkit.Chem.AllChem.MMFFOptimizeMolecule","rdkit.Chem.AllChem.UFFOptimizeMolecule")
        self._to_xyz_core_docs =(
            "http://rdkit.org/docs/source/rdkit.Chem.rdForceFieldHelpers.html?highlight=mmff#rdkit.Chem.rdForceFieldHelpers.MMFFOptimizeMolecule",
            "http://rdkit.org/docs/source/rdkit.Chem.rdForceFieldHelpers.html?highlight=mmff#rdkit.Chem.rdForceFieldHelpers.UFFOptimizeMolecule"
        )

    @property
    def smiles(self):
        return self._smiles

    @property
    def smiles_args(self):
        return self._smiles_args

    @property
    def smarts(self):
        return self._smarts

    @property
    def smarts_args(self):
        return self._smarts_args

    @property
    def inchi(self):
        return self._inchi

    @property
    def inchi_args(self):
        return self._inchi_args

    @property
    def xyz(self):
        return self._xyz

    @property
    def UFF_args(self):
        return self._UFF_args

    @property
    def MMFF_args(self):
        return self._MMFF_args

    def _check_rdkit_molecule(self):
        if self.rdkit_molecule is None:
            raise ValueError('The molecule object has not been created yet.')

    def hydrogens(self, action='add', **kwargs):
        """
        This function adds or removes to an rdkit molecule.

        Parameters
        ----------
        action: str
            Either 'add' or 'remove', to add hydrogns or remove them from the rdkit molecule.

        kwargs:
            The arguments that can be passed to the rdkit functions:
            - `Chem.AddHs`: documentation at http://rdkit.org/docs/source/rdkit.Chem.rdmolops.html?highlight=addhs#rdkit.Chem.rdmolops.AddHs
            - `Chem.RemoveHs`: documentation at http://rdkit.org/docs/source/rdkit.Chem.rdmolops.html?highlight=addhs#rdkit.Chem.rdmolops.RemoveHs

        Notes
        -----
            - The molecule is modified in place.
            - The rdkit molecule object must be created in advance.

        """

        # rdkit molecule must exist
        self._check_rdkit_molecule()

        if action == 'add':
            self.rdkit_molecule = Chem.AddHs(self.rdkit_molecule, **kwargs)
        elif action == 'remove':
            self.rdkit_molecule = Chem.RemoveHs(self.rdkit_molecule, **kwargs)
        else:
            raise ValueError("The parameter 'action' must be either of 'add' or 'remove'.")

    def load(self, input, input_type, **kwargs):
        """

        Parameters
        ----------
        input: str
            The representation string or path to a file.

        input_type: str
            The input type. The legit types are enlisted here:
                - smiles: The input must be SMILES representation of a molecule.
                - smarts: The input must be SMARTS representation of a molecule.
                - inchi: The input must be InChi representation of a molecule.

        kwargs:
            The corresponding RDKit arguments for each of the input types:
                - smiles: http://rdkit.org/docs/source/rdkit.Chem.rdmolfiles.html#rdkit.Chem.rdmolfiles.MolFromSmiles
                - smarts: http://rdkit.org/docs/source/rdkit.Chem.rdmolfiles.html#rdkit.Chem.rdmolfiles.MolFromSmarts
                - inchi: http://rdkit.org/docs/source/rdkit.Chem.inchi.html?highlight=inchi#rdkit.Chem.inchi.MolFromInchi

        Notes
        -----
            - The molecule will be creatd as an RDKit molecule object.
            - The molecule object will be stored and available as `rdkit_molecule` attribute.
            - If you load a molecule from its SMARTS string, there is high probability that you can't convert it to other
                types due to the abstract description of the molecules by SMARTS representation.

        """
        if input_type == 'smiles':
            creator = ('SMILES', input)
            mol = Chem.MolFromSmiles(input, **kwargs)
        elif input_type == 'smarts':
            creator = ('SMARTS', input)
            mol = Chem.MolFromSmarts(input, **kwargs)
        elif input_type == 'inchi':
            creator = ('InChi', input)
            mol = Chem.MolFromInchi(input, **kwargs)
        else:
            msg = "The input type is not legit or has not been implemented yet."
            raise ValueError(msg)

        # check for None values. Some error messages that may have already raised:
        # [16:59:24] SMILES Parse Error: syntax error for input: 'NotSMILES'
        # [12:18:01] Explicit valence for atom # 1 O greater than permitted
        # [12:20:41] Can't kekulize mol
        if mol is None:
            raise ValueError('The input is not a legit %s string. Refere to the error message that was already displayed by RDKit library!'%creator[0])

        # if the molecule is already being created warn the user
        if self.rdkit_molecule and self.creator:
            msg = "The molecule was already built using %s, is now overwritten by %s" % (
            str(self.creator), str(creator))
            warnings.warn(msg)

        self.creator = creator
        self.rdkit_molecule = mol

        self._init_attributes()

        if self.creator[0]=='SMILES':
            self.to_smiles()
        elif self.creator[0]=='SMARTS':
            self.to_smarts()
        elif self.creator[0]=='InChi':
            self.to_inchi()

    def to_smiles(self, **kwargs):
        """
        This function creates and stores the SMILES string for a pre-built molecule.

        Parameters
        ----------
        kwargs:
            The arguments that can be passed to the rdkit.Chem.MolToSmiles function.
            The documentation is available at: http://rdkit.org/docs/source/rdkit.Chem.rdmolfiles.html#rdkit.Chem.rdmolfiles.MolToSmiles

        Notes
        -----
            - The molecule is modified in place.
            - The rdkit molecule object must be created in advance.
            - The SMILES string is canocical by default, unless when one requests kekuleSmiles.

        """
        # rdkit molecule must exist
        self._check_rdkit_molecule()

        # kekulize flag
        if 'kekuleSmiles' in kwargs and kwargs['kekuleSmiles']:
            Chem.Kekulize(self.rdkit_molecule)

        # store arguments for future reference
        self._smiles = Chem.MolToSmiles(self.rdkit_molecule, **kwargs)

        # arguments
        self._default_smiles_args = {"isomericSmiles":True, "kekuleSmiles":False, "rootedAtAtom":-1, "canonical":True, "allBondsExplicit":False,
                "allHsExplicit":False, "doRandom":False}
        self._smiles_args = update_default_kwargs(self._default_smiles_args, kwargs,
                                                 self._to_smiles_core_names[0], self._to_smiles_core_docs[0])

    def to_smarts(self, **kwargs):
        """
        This function creates and stores the SMARTS string for a pre-built molecule.

        Parameters
        ----------
        kwargs:
            All the arguments that can be passed to the rdkit.Chem.MolToSmarts function.
            The documentation is available at: http://rdkit.org/docs/source/rdkit.Chem.rdmolfiles.html#rdkit.Chem.rdmolfiles.MolToSmarts

        Notes
        -----
            - The molecule is modified in place.
            - The rdkit molecule object must be created in advance.

        """
        # rdkit molecule must exist
        self._check_rdkit_molecule()


        # store arguments for future reference
        self._smarts = Chem.MolToSmarts(self.rdkit_molecule, **kwargs)

        # arguments
        self._default_smarts_args = {"isomericSmiles":True}
        self._smarts_args = update_default_kwargs(self._default_smarts_args, kwargs,
                                                 self._to_smarts_core_names[0], self._to_smarts_core_docs[0])

    def to_inchi(self, **kwargs):
        """
        This function creates and stores the InChi string for a pre-built molecule.

        Parameters
        ----------
        kwargs:
            All the arguments that can be passed to the rdkit.Chem.MolToInchi function.
            The documentation is available at: http://rdkit.org/docs/source/rdkit.Chem.inchi.html?highlight=inchi#rdkit.Chem.inchi.MolToInchi

        Notes
        -----
            - The molecule is modified in place.
            - The rdkit molecule object must be created in advance.

        """
        # rdkit molecule must exist
        self._check_rdkit_molecule()

        # store arguments for future reference
        self._inchi = Chem.MolToInchi(self.rdkit_molecule, **kwargs)

        # arguments
        self._default_inchi_args = {"options":'', 'logLevel':None, 'treatWarningAsError':False}
        self._inchi_args = update_default_kwargs(self._default_inchi_args, kwargs,
                                                 self._to_inchi_core_names[0], self._to_inchi_core_docs[0])

    def to_xyz(self, optimizer=None, **kwargs):
        """
        This function creates and stores the xyz coordinates for a pre-built molecule object.

        Parameters
        ----------
        optimizer: None or str, optional (default: None)
            If None, the geometries will be extracted from the passed conformer.
            Otherwise, any of the 'UFF' or 'MMFF' force fileds should be passed to embed and optimize geometries using 'rdkit.Chem.AllChem.UFFOptimizeMolecule' or
            'rdkit.Chem.AllChem.MMFFOptimizeMolecule' methods, respectively.

        kwargs:
            The arguments that can be passed to the corresponding rdkit functions.
            The documentation is available at:
                - UFFOptimizeMolecule: http://rdkit.org/docs/source/rdkit.Chem.rdForceFieldHelpers.html?highlight=mmff#rdkit.Chem.rdForceFieldHelpers.UFFOptimizeMolecule
                - MMFFOptimizeMolecule: http://rdkit.org/docs/source/rdkit.Chem.rdForceFieldHelpers.html?highlight=mmff#rdkit.Chem.rdForceFieldHelpers.MMFFOptimizeMolecule

        Notes
        -----
            - The geometry will be stored in the `xyz` attribute.
            - The rdkit molecule object must be created in advance.
            - The hydrogens won't be added to the molecule automatically. You should magane that using `hydrogens` method in advance.
            - If the molecule object has been built using 2D representations (e.g., SMILES or InChi), the conformer
            doesn't exist and you nedd to set the optimizer parameter to any of the force fields.


        """
        # rdkit molecule must exist
        self._check_rdkit_molecule()

        # add hydrogens
        # self.hydrogens('add')

        # embeding and optimization
        if optimizer:
            AllChem.EmbedMolecule(self.rdkit_molecule)
            if optimizer ==  'MMFF':
                if AllChem.MMFFHasAllMoleculeParams(self.rdkit_molecule):
                    AllChem.MMFFOptimizeMolecule(self.rdkit_molecule, **kwargs)
                else:
                    msg = "The MMFF parameters are not available for all of the molecule’s atoms."
                    raise ValueError(msg)
            elif optimizer == 'UFF':
                if AllChem.UFFHasAllMoleculeParams(self.rdkit_molecule):
                    AllChem.UFFOptimizeMolecule(self.rdkit_molecule, **kwargs)
                else:
                    msg = "The UFF parameters are not available for all of the molecule’s atoms."
                    raise ValueError(msg)
            else:
                msg = "The '%s' is not a legit value for the optimizer parameter."%str(optimizer)
                raise ValueError(msg)

        # get conformer
        try:
            conf = self.rdkit_molecule.GetConformer()
        except ValueError:
            msg = "The conformation has not been built yet. Maybe due to the 2D representation of the creator.\n" \
                  "You should set the optimizer value if you wish to embed and optimize the 3D geometries."
            raise ValueError(msg)
        geometry = conf.GetPositions()
        atoms_list = self.rdkit_molecule.GetAtoms()
        atomic_nums = np.array([i.GetAtomicNum() for i in atoms_list])
        atomic_symbols = np.array([i.GetSymbol() for i in atoms_list])
        self._xyz = XYZ(geometry, atomic_nums.reshape(-1,1), atomic_symbols.reshape(-1,1))

        # arguments
        self._default_UFF_args = {'maxIters':200, 'vdwThresh':10.0, 'confId':-1, 'ignoreInterfragInteractions':True}
        self._default_MMFF_args = {'mmffVariant':'MMFF94', 'maxIters':200, 'nonBondedThresh':100.0, 'confId':-1, 'ignoreInterfragInteractions':True}

        if optimizer=='UFF':
            self._UFF_args = update_default_kwargs(self._default_UFF_args, kwargs,
                                                 self._to_xyz_core_names[1], self._to_xyz_core_docs[1])
        elif optimizer=='MMFF':
            self._MMFF_args = update_default_kwargs(self._default_MMFF_args, kwargs,
                                                 self._to_xyz_core_names[0], self._to_xyz_core_docs[0])