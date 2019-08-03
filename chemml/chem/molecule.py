from __future__ import print_function
import os
import keras # required to go around the protobuf error after importing pybel prior to tensorflow
from rdkit import Chem
import pybel
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

    def __repr__(self):
        return '<XYZ(geometry: {self.geometry.shape!r},' \
               ' atomic_numbers: {self.atomic_numbers.shape!r},' \
               ' atomic_symbols: {self.atomic_symbols.shape!r})>'.format(self=self)


class Molecule(object):
    """
    The central class to construct a molecule from different chemical input formats.
    This module is built on top of RDKit and OpenBabel python API.
    We join the forces and strength of these two cheminformatic libraris for a consistent user experience.

    Almost all the molecular descriptors and molecule-based ML models require the chemical informatin as a Molecule object.
    Several methods are available in this module to facilitate the manipulation of chemical data.

    Parameters
    ----------
    input: str
        The representation string or path to a file.

    input_type: str
        The input type.
        The available types are enlisted here:
            - smiles: The input must be SMILES representation of a molecule.
            - smarts: The input must be SMARTS representation of a molecule.
            - inchi: The input must be InChi representation of a molecule.
            - xyz:  The input must be the path to an xyz file.

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
        The class object that stores the 3D info. The available attributes in the class are 'geometry',
        'atomic_numbers', and 'atomic_symbols'.

    Examples
    --------
    >>> from chemml.chem import Molecule
    >>> caffeine_smiles = 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'
    >>> caffeine_smarts = '[#6]-[#7]1:[#6]:[#7]:[#6]2:[#6]:1:[#6](=[#8]):[#7](:[#6](=[#8]):[#7]:2-[#6])-[#6]'
    >>> caffeine_inchi = 'InChI=1S/C8H10N4O2/c1-10-4-9-6-5(10)7(13)12(3)8(14)11(6)2/h4H,1-3H3'
    >>> mol = Molecule(caffeine_smiles, 'smiles')
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
    def __init__(self, input, input_type, **kwargs):
        self.rdkit_molecule = None
        self.pybel_molecule = None
        self.creator = None
        self._init_attributes()
        self._extra_docs()
        self._load(input, input_type, **kwargs)

    def __repr__(self):
        return '<chemml.chem.Molecule(\n' \
               '        rdkit_molecule : {self.rdkit_molecule!r},\n' \
               '        pybel_molecule : {self.pybel_molecule!r},\n' \
               '        creator        : {self.creator!r},\n' \
               '        smiles         : {self.smiles!r},\n' \
               '        smarts         : {self.smarts!r},\n' \
               '        inchi          : {self.inchi!r},\n' \
               '        xyz            : {self.xyz!r})>'.format(self=self)

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
        # default arguments
        self._default_rdkit_smiles_args = {"isomericSmiles":True, "kekuleSmiles":False, "rootedAtAtom":-1, "canonical":True, "allBondsExplicit":False,
                "allHsExplicit":False} #"doRandom":False
        self._default_rdkit_smarts_args = {"isomericSmiles":True}
        self._default_rdkit_inchi_args = {"options":'', 'logLevel':None, 'treatWarningAsError':False}
        self._default_UFF_args = {'maxIters':200, 'vdwThresh':10.0, 'confId':-1, 'ignoreInterfragInteractions':True}
        self._default_MMFF_args = {'mmffVariant':'MMFF94', 'maxIters':200, 'nonBondedThresh':100.0, 'confId':-1, 'ignoreInterfragInteractions':True}

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

    def _check_original_molecule(self):
        if self.rdkit_molecule is None and self.pybel_molecule is None:
            msg = "Neither rdkit nor pybel molecule object has been created yet."
            raise ValueError(msg)
        elif self.rdkit_molecule is None:
            return 'pybel'
        elif self.pybel_molecule is None:
            return 'rdkit'
        # in case both of them are available go with rdkit
        else:
            return 'rdkit'

    def hydrogens(self, action='add', **kwargs):
        """
        This function adds/removes hydrogens to/from a prebuilt molecule object.

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
            - The rdkit or pybel molecule object must be created in advance.
            - Only rdkit or pybel molecule object will be modified in place.
            - If you remove hydrogens from molecules, the atomic 3D coordinates might not be accurate for the conversion to xyz representation.

        """

        # molecule must exist
        _ = self._check_original_molecule()

        if action == 'add':
            if self.pybel_molecule:
                self.pybel_molecule.addh()
            # Note: just if not elif
            if self.rdkit_molecule:
                self.rdkit_molecule = Chem.AddHs(self.rdkit_molecule, **kwargs)
        elif action == 'remove':
            if self.pybel_molecule:
                self.pybel_molecule.removeh()
            if self.rdkit_molecule:
                self.rdkit_molecule = Chem.RemoveHs(self.rdkit_molecule, **kwargs)
        else:
            raise ValueError("The parameter 'action' must be either of 'add' or 'remove'.")

    def _load(self, input, input_type, **kwargs):
        """
        The main internal function to load a molecule.
        """
        if input_type == 'xyz':
            self._load_pybel(input, input_type, **kwargs)
        elif input_type in ['smiles', 'smarts', 'inchi']:
            self._load_rdkit(input, input_type, **kwargs)
        else:
            msg = "The input type '%s' is not supported." %input_type
            raise ValueError(msg)

    def _load_rdkit(self, input, input_type, from_load=True, **kwargs):
        """
        The internal function to load a molecule using rdkit engine.
        """

        if input_type == 'smiles':
            creator = ('SMILES', input)
            rdkit_mol = Chem.MolFromSmiles(input, **kwargs)
        elif input_type == 'smarts':
            creator = ('SMARTS', input)
            rdkit_mol = Chem.MolFromSmarts(input, **kwargs)
        elif input_type == 'inchi':
            creator = ('InChi', input)
            rdkit_mol = Chem.MolFromInchi(input, **kwargs)
        else:
            msg = "The input type '%s' is not available from 'rdkit' engine." % (input_type)
            raise ValueError(msg)

        # check for None values. Some error messages that may have already raised:
        # [16:59:24] SMILES Parse Error: syntax error for input: 'NotSMILES'
        # [12:18:01] Explicit valence for atom # 1 O greater than permitted
        # [12:20:41] Can't kekulize mol
        if rdkit_mol is None:
            msg = 'The input is not a legit %s string. Refere to the error message that was already displayed by RDKit library!'%creator[0]
            raise ValueError(msg)

        # if the molecule is already being created warn the user
        if self.rdkit_molecule and self.creator:
            msg = "The molecule was already built using %s, is now overwritten by %s" % (
            str(self.creator), str(creator))
            warnings.warn(msg)

        self.rdkit_molecule = rdkit_mol

        if from_load:
            self.creator = creator
            self._init_attributes()

        if self.creator[0]=='SMILES':
            self.to_smiles()
        elif self.creator[0]=='SMARTS':
            self.to_smarts()
        elif self.creator[0]=='InChi':
            self.to_inchi()

    def _load_pybel(self, input, input_type, **kwargs):
        """
        The internal function to load a molecule using rdkit engine.

        """
        # available variables
        pybel_mol = None
        creator = None

        if input_type == 'xyz':
            if os.path.isfile(input):
                creator = ('XYZ', input)
                gen = pybel.readfile("xyz", input)
                mols = list(gen)
                if len(mols) == 1:
                    pybel_mol = mols[0]
                else:
                    warnings.warn('More than one Molecule object is created and is returned as a generator.')
                    return self._multiple_molecules(mols, creator)
            else:
                msg = "The input '%s' is not a valid XYZ input file."%input
                raise ValueError(msg)

        if pybel_mol is None:
            msg = 'The input is not a legit %s string. Refere to the error message that was already displayed by Pybel library!' % \
              creator[0]
            raise ValueError(msg)

        # if the molecule is already being created warn the user
        if self.pybel_molecule and self.creator:
            msg = "The molecule was already built using %s, is now overwritten by %s" % (
            str(self.creator), str(creator))
            warnings.warn(msg)

        self.creator = creator
        self.pybel_molecule = pybel_mol

        self._init_attributes()

        if self.creator[0]=='XYZ':
            self.to_xyz()

    def _multiple_molecules(self, mols, creator):
        for mol in mols:
            m = Molecule()
            m.pybel_molecule = mol
            m.creator = creator
            yield m

    def to_smiles(self, **kwargs):
        """
        This function creates and stores the SMILES string for a pre-built molecule.

        Parameters
        ----------
        kwargs:
            The arguments for the rdkit.Chem.MolToSmiles function.
            The documentation is available at: http://rdkit.org/docs/source/rdkit.Chem.rdmolfiles.html#rdkit.Chem.rdmolfiles.MolToSmiles

        Notes
        -----
            - The rdkit or pybel molecule object must be created in advance.
            - If only pybel molecule is available, we create an rdkit molecule using its SMILES representation, and then recreate the SMILES string using rdkit arguments.
            - The molecule will be modified in place.
            - For rdkit molecule the SMILES string is canocical by default, unless when one requests kekuleSmiles.

        """
        # molecule must exist
        engine = self._check_original_molecule()
        if engine == 'pybel':
            smiles = self.pybel_molecule.write('smi').strip().split('\t')[0]
            self._load_rdkit(smiles, 'smiles', from_load=False)
            self.to_smiles()
        else:
            self._to_smiles_rdkit(**kwargs)

    def _to_smiles_rdkit(self, **kwargs):
        """
        This internal function creates and stores the SMILES string for rdkit molecule.
        """
        # kekulize flag
        if 'kekuleSmiles' in kwargs and kwargs['kekuleSmiles']:
            Chem.Kekulize(self.rdkit_molecule)

        # store arguments for future reference
        self._smiles = Chem.MolToSmiles(self.rdkit_molecule, **kwargs)

        # arguments
        self._smiles_args = update_default_kwargs(self._default_rdkit_smiles_args, kwargs,
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
            - The rdkit or pybel molecule object must be created in advance.
            - If only pybel molecule is available, we create an rdkit molecule using its SMILES representation, and then create the SMARTS string using rdkit arguments.
            - The molecule will be modified in place.

        """
        # molecule must exist
        engine = self._check_original_molecule()

        if engine == 'rdkit':
            self._to_smarts_rdkit(**kwargs)
        else:
            smiles = self.pybel_molecule.write('smi').strip().split('\t')[0]
            self._load_rdkit(smiles, 'smiles', from_load=False)
            self._to_smarts_rdkit(**kwargs)

    def _to_smarts_rdkit(self, **kwargs):
        """
        The internal
        """
        # store arguments for future reference
        self._smarts = Chem.MolToSmarts(self.rdkit_molecule, **kwargs)

        # arguments
        self._smarts_args = update_default_kwargs(self._default_rdkit_smarts_args, kwargs,
                                                 self._to_smarts_core_names[0], self._to_smarts_core_docs[0])

    def to_inchi(self, **kwargs):
        """
        This function creates and stores the InChi string for a pre-built molecule.

        Parameters
        ----------
        kwargs:
            The arguments that can be passed to the rdkit.Chem.MolToInchi function (will be used only if rdkit molecule is available).
            The documentation is available at: http://rdkit.org/docs/source/rdkit.Chem.inchi.html?highlight=inchi#rdkit.Chem.inchi.MolToInchi

        Notes
        -----
            - The rdkit or pybel molecule object must be created in advance.
            - The molecule will be modified in place.

        """
        # molecule must exist
        engine = self._check_original_molecule()
        if engine == 'pybel':
            self._to_inchi_pybel()
        else:
            self._to_inchi_rdkit(**kwargs)

    def _to_inchi_rdkit(self, **kwargs):
        """
        This internal function creates and stores the InChi string for rdkit molecule.
        """
        # store arguments for future reference
        self._inchi = Chem.MolToInchi(self.rdkit_molecule, **kwargs)

        # arguments
        self._inchi_args = update_default_kwargs(self._default_rdkit_inchi_args, kwargs,
                                                 self._to_inchi_core_names[0], self._to_inchi_core_docs[0])

    def _to_inchi_pybel(self):
        """
        This internal function creates and stores the InChi string for pybel molecule.
        """
        self._inchi = self.pybel_molecule.write('inchi').strip()

    def to_xyz(self, optimizer=None, **kwargs):
        """
        This function creates and stores the xyz coordinates for a pre-built molecule object.

        Parameters
        ----------
        optimizer: None or str, optional (default: None)
            If None, the geometries will be extracted from the available source of 3D structure (if any).
            Otherwise, any of the 'UFF' or 'MMFF' force fileds should be passed to embed and optimize geometries using 'rdkit.Chem.AllChem.UFFOptimizeMolecule' or
            'rdkit.Chem.AllChem.MMFFOptimizeMolecule' methods, respectively.

        kwargs:
            The arguments that can be passed to the corresponding forcefileds.
            The documentation is available at:
                - UFFOptimizeMolecule: http://rdkit.org/docs/source/rdkit.Chem.rdForceFieldHelpers.html?highlight=mmff#rdkit.Chem.rdForceFieldHelpers.UFFOptimizeMolecule
                - MMFFOptimizeMolecule: http://rdkit.org/docs/source/rdkit.Chem.rdForceFieldHelpers.html?highlight=mmff#rdkit.Chem.rdForceFieldHelpers.MMFFOptimizeMolecule

        Notes
        -----
            - The geometry will be stored in the `xyz` attribute.
            - The molecule object must be created in advance.
            - The hydrogens won't be added to the molecule automatically. You should add it manually using `hydrogens` method.
            - If the molecule object has been built using 2D representations (e.g., SMILES or InChi), the conformer
            doesn't exist and you nedd to set the optimizer parameter to any of the force fields.
            - If the 3D info exist but you still need to run optimization, the 3D structure will be embedded from scratch (i.e., the current atom coordinates will be removed.)


        """
        # rdkit molecule must exist
        engine = self._check_original_molecule()
        # any futher process depends on the state of the optimizer and available 3D info
        if self.xyz and optimizer is None:
            return True     # the required info is available
        elif self.pybel_molecule:
            if optimizer is None :
                self._to_xyz_pybel()    # pybel molecule always contains the 3D info
            else:
                # build the rdkit molecule from 2D info
                smiles = self.pybel_molecule.write('smi').strip().split('\t')[0]
                self._load_rdkit(smiles, 'smiles', from_load=False)
                self.creator = ('SMILES', smiles)
                self.hydrogens('add')
                self._to_xyz_rdkit(optimizer, **kwargs)
        elif engine == 'rdkit':
            self._to_xyz_rdkit(optimizer, **kwargs)

    def _to_xyz_rdkit(self, optimizer, **kwargs):
        """
        The internal function creates and stores the xyz coordinates for a pre-built molecule object.
        """
        # add hydrogens >> commented out and left for the users to take care of it using hydrogens method.
        # self.hydrogens('add')

        # embeding and optimization
        if optimizer:
            AllChem.EmbedMolecule(self.rdkit_molecule)
            if optimizer ==  'MMFF':
                if AllChem.MMFFHasAllMoleculeParams(self.rdkit_molecule):
                    AllChem.MMFFOptimizeMolecule(self.rdkit_molecule, **kwargs)
                else:
                    msg = "The MMFF parameters are not available for all of the molecule's atoms."
                    raise ValueError(msg)
            elif optimizer == 'UFF':
                if AllChem.UFFHasAllMoleculeParams(self.rdkit_molecule):
                    AllChem.UFFOptimizeMolecule(self.rdkit_molecule, **kwargs)
                else:
                    msg = "The UFF parameters are not available for all of the molecule's atoms."
                    raise ValueError(msg)
            else:
                msg = "The '%s' is not a legit value for the optimizer parameter."%str(optimizer)
                raise ValueError(msg)

        # get conformer
        try:
            conf = self.rdkit_molecule.GetConformer()
        except ValueError:
            msg = "The conformation has not been built yet (maybe due to the 2D representation of the creator).\n" \
                  "You should set the optimizer value if you wish to embed and optimize the 3D geometry."
            raise ValueError(msg)
        geometry = conf.GetPositions()
        atoms_list = self.rdkit_molecule.GetAtoms()
        atomic_nums = np.array([i.GetAtomicNum() for i in atoms_list])
        atomic_symbols = np.array([i.GetSymbol() for i in atoms_list])
        self._xyz = XYZ(geometry, atomic_nums.reshape(-1,1), atomic_symbols.reshape(-1,1))

        if optimizer=='UFF':
            self._UFF_args = update_default_kwargs(self._default_UFF_args, kwargs,
                                                 self._to_xyz_core_names[1], self._to_xyz_core_docs[1])
        elif optimizer=='MMFF':
            self._MMFF_args = update_default_kwargs(self._default_MMFF_args, kwargs,
                                                 self._to_xyz_core_names[0], self._to_xyz_core_docs[0])

    def _to_xyz_pybel(self):
        """
        The internal function extracts the xyz coordinates from a pre-built pybel molecule object.
        """
        geometry = np.array([atom.coords for atom in self.pybel_molecule])
        atomic_nums = np.array([atom.atomicnum for atom in self.pybel_molecule])
        # pybel atom types are extended
        Z = {44: 'Ru',
             75: 'Re',
             104: 'Rf',
             111: 'Rg',
             88: 'Ra',
             37: 'Rb',
             86: 'Rn',
             45: 'Rh',
             4: 'Be',
             56: 'Ba',
             107: 'Bh',
             83: 'Bi',
             97: 'Bk',
             35: 'Br',
             1: 'H',
             15: 'P',
             76: 'Os',
             99: 'Es',
             80: 'Hg',
             32: 'Ge',
             64: 'Gd',
             31: 'Ga',
             59: 'Pr',
             78: 'Pt',
             94: 'Pu',
             6: 'C',
             82: 'Pb',
             91: 'Pa',
             46: 'Pd',
             48: 'Cd',
             84: 'Po',
             61: 'Pm',
             108: 'Hs',
             115: 'Uup',
             117: 'Uus',
             118: 'Uuo',
             67: 'Ho',
             72: 'Hf',
             19: 'K',
             2: 'He',
             101: 'Md',
             12: 'Mg',
             42: 'Mo',
             25: 'Mn',
             8: 'O',
             109: 'Mt',
             16: 'S',
             74: 'W',
             30: 'Zn',
             63: 'Eu',
             40: 'Zr',
             68: 'Er',
             28: 'Ni',
             102: 'No',
             11: 'Na',
             41: 'Nb',
             60: 'Nd',
             10: 'Ne',
             93: 'Np',
             87: 'Fr',
             26: 'Fe',
             114: 'Fl',
             100: 'Fm',
             5: 'B',
             9: 'F',
             38: 'Sr',
             7: 'N',
             36: 'Kr',
             14: 'Si',
             50: 'Sn',
             62: 'Sm',
             23: 'V',
             21: 'Sc',
             51: 'Sb',
             106: 'Sg',
             34: 'Se',
             27: 'Co',
             112: 'Cn',
             96: 'Cm',
             17: 'Cl',
             20: 'Ca',
             98: 'Cf',
             58: 'Ce',
             54: 'Xe',
             71: 'Lu',
             55: 'Cs',
             24: 'Cr',
             29: 'Cu',
             57: 'La',
             3: 'Li',
             116: 'Lv',
             81: 'Tl',
             69: 'Tm',
             103: 'Lr',
             90: 'Th',
             22: 'Ti',
             52: 'Te',
             65: 'Tb',
             43: 'Tc',
             73: 'Ta',
             70: 'Yb',
             105: 'Db',
             66: 'Dy',
             110: 'Ds',
             53: 'I',
             92: 'U',
             39: 'Y',
             89: 'Ac',
             47: 'Ag',
             113: 'Uut',
             77: 'Ir',
             95: 'Am',
             13: 'Al',
             33: 'As',
             18: 'Ar',
             79: 'Au',
             85: 'At',
             49: 'In'}
        atomic_symbols = np.array([Z[i] for i in atomic_nums])
        self._xyz = XYZ(geometry, atomic_nums.reshape(-1, 1), atomic_symbols.reshape(-1, 1))

    def visualize(self, filename=None, **kwargs):
        """
        This function visualizes the molecule. If both rdkit and pybel objects are avaialble, the rdkit object
        will be used for visualization.

        Parameters
        ----------
        filename: str, optional (default = None)
            This is the path to the file that you want write the image in it.
            Tkinter and Python Imaging Library are required for writing the image.

        kwargs:
            any extra parameter that you want to pass to the rdkit or pybel draw tool.
            Additional information at:
                - https://www.rdkit.org/docs/source/rdkit.Chem.Draw.html
                - http://openbabel.org/docs/dev/UseTheLibrary/Python_PybelAPI.html#pybel.Molecule.draw

        Returns
        -------
        object
            You will be able to display this object, e.g., inside the Jupyter Notebook.

        """
        engine = self._check_original_molecule()
        if engine == 'rdkit':
            from rdkit.Chem import Draw
            if filename is not None:
                Draw.MolToFile(self.rdkit_molecule, filename, **kwargs)
            else:
                return Draw.MolToImage(self.rdkit_molecule, **kwargs)
        elif engine == 'pybel':
            if filename is not None:
                self.pybel_molecule.draw(show=False, filename=filename, **kwargs)
            else:
                return self.pybel_molecule # it seems that the object alone is displayable

