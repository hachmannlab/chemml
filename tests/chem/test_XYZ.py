import pytest
from rdkit import Chem
import warnings

from chemml.chem import Molecule


@pytest.fixture()
def caffeine_smiles():
    smiles = 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'
    return smiles

@pytest.fixture()
def caffeine_canonical():
    smiles = 'Cn1c(=O)c2c(ncn2C)n(C)c1=O'
    return smiles

@pytest.fixture()
def caffeine_kekulize():
    smiles = 'CN1C(=O)C2=C(N=CN2C)N(C)C1=O'
    return smiles

@pytest.fixture()
def caffeine_smarts():
    smarts = '[#6]-[#7]1:[#6]:[#7]:[#6]2:[#6]:1:[#6](=[#8]):[#7](:[#6](=[#8]):[#7]:2-[#6])-[#6]'
    return smarts

@pytest.fixture()
def caffeine_inchi():
    inchi = 'InChI=1S/C8H10N4O2/c1-10-4-9-6-5(10)7(13)12(3)8(14)11(6)2/h4H,1-3H3'
    return inchi

def test_exception_smiles():
    m = Molecule()
    # syntax error: SMILES Parse Error
    with pytest.raises(ValueError):
        m.load('fake', 'smiles')
    # wrong SMILES
    with pytest.raises(ValueError):
        m.load('CO(C)C', 'smiles')
    # can't kekulize
    with pytest.raises(ValueError):
        m.load('c1cc1', 'smiles')

def test_exception_smarts():
    m = Molecule()
    # syntax error: SMARTS Parse Error
    with pytest.raises(ValueError):
        m.load('fake', 'smarts')
    # syntax error: SMARTS Parse Error
    with pytest.raises(ValueError):
        m.load('[#6](=[#6]', 'smarts')

def test_exception_inchi():
    m = Molecule()
    # syntax error: SMARTS Parse Error
    with pytest.raises(ValueError):
        m.load('fake', 'inchi')
    # inchi key: syntax error: SMARTS Parse Error
    with pytest.raises(ValueError):
        m.load('RYYVLZVUVIJVGH-UHFFFAOYSA-N', 'inchi')

def test_load_smiles(caffeine_smiles):
    m = Molecule()
    m.load(caffeine_smiles, 'smiles')
    assert m.creator == ('SMILES', caffeine_smiles)
    assert isinstance(m.rdkit_molecule, Chem.Mol)
    assert m.smiles is None

def test_reload_warning(caffeine_smiles):
    m = Molecule()
    m.load(caffeine_smiles, 'smiles')
    warnings.simplefilter("always")
    with warnings.catch_warnings(record=True) as w:
        m.load(caffeine_smiles, 'smiles')
        assert m.smiles is None

def test_load_smarts(caffeine_smarts):
    m = Molecule()
    m.load(caffeine_smarts, 'smarts')
    assert m.creator == ('SMARTS', caffeine_smarts)
    assert isinstance(m.rdkit_molecule, Chem.Mol)
    assert m.smarts is None

def test_load_inchi(caffeine_inchi):
    m = Molecule()
    m.load(caffeine_inchi, 'inchi')
    assert m.creator == ('InChi', caffeine_inchi)
    assert isinstance(m.rdkit_molecule, Chem.Mol)
    assert m.inchi is None

def test_mol2smiles(caffeine_smiles, caffeine_canonical, caffeine_kekulize):
    m = Molecule()
    m.load(caffeine_smiles, 'smiles')
    m.to_smiles()
    assert isinstance(m.rdkit_molecule, Chem.Mol)
    assert m.smiles == caffeine_canonical
    m.to_smiles(kekuleSmiles=True)
    assert m.smiles == caffeine_kekulize
    # kekulize takes priority over canonical
    m.to_smiles(kekuleSmiles=True)
    assert m.smiles == caffeine_kekulize
    # if kekule is False, everything is OK
    m.to_smiles(canonical=True)
    assert m.smiles == caffeine_canonical

def test_mol2smiles_defaultargs(caffeine_smiles):
    m = Molecule()
    m.load(caffeine_smiles, 'smiles')
    m.to_smiles()
    m.to_smiles(**m._default_smiles_args)
    # all the args & canonical=False & doRandom=True
    args = m._default_smiles_args
    args['canonical'] = False
    args['doRandom'] = True
    m.to_smiles(**args)
    assert m.smiles_args['canonical'] is False
    assert m.smiles_args['doRandom'] is True

def test_mol2smarts(caffeine_smarts):
    m = Molecule()
    m.load(caffeine_smarts, 'smarts')
    m.to_smarts()
    assert isinstance(m.rdkit_molecule, Chem.Mol)
    assert m.smarts == caffeine_smarts
    m.to_smarts(isomericSmiles=False)
    assert m.smarts == caffeine_smarts

def test_mol2smarts_defaultargs(caffeine_smarts):
    m = Molecule()
    m.load(caffeine_smarts, 'smarts')
    m.to_smarts()
    m.to_smarts(**m._default_smarts_args)
    # all the args & isomericSmiles=False
    args = m._default_smarts_args
    args['isomericSmiles'] = False
    m.to_smarts(**args)
    assert m.smarts_args['isomericSmiles'] is False

def test_mol2inchi(caffeine_inchi):
    m = Molecule()
    m.load(caffeine_inchi, 'inchi')
    m.to_inchi()
    assert isinstance(m.rdkit_molecule, Chem.Mol)
    assert m.inchi == caffeine_inchi

def test_mol2inchi_defaultargs(caffeine_inchi):
    m = Molecule()
    m.load(caffeine_inchi, 'inchi')
    m.to_inchi()
    m.to_inchi(**m._default_inchi_args)
    # all the args & treatWarningAsError=True
    args = m._default_inchi_args
    args['treatWarningAsError'] = True
    m.to_inchi(**args)
    assert m.inchi_args['treatWarningAsError'] is True

def test_hydrogens(caffeine_smiles, caffeine_canonical, caffeine_inchi):
    m = Molecule()
    m.load(caffeine_smiles, 'smiles')
    m.to_smiles()
    assert isinstance(m.rdkit_molecule, Chem.Mol)
    assert m.smiles == caffeine_canonical
    # add hydrogens
    m.hydrogens('add', explicitOnly=False)
    # canonical smiles with hydrogens
    m.to_smiles()
    assert m.smiles == "[H]c1nc2c(c(=O)n(C([H])([H])[H])c(=O)n2C([H])([H])[H])n1C([H])([H])[H]"
    # test inchi
    m = Molecule()
    m.load(caffeine_inchi, 'inchi')
    m.to_inchi()
    assert m.inchi == caffeine_inchi
    # add hydrogens
    m.hydrogens('add')
    m.to_inchi()
    assert m.inchi == caffeine_inchi

def test_hydrogens_exception(caffeine_smiles):
    m = Molecule()
    m.load(caffeine_smiles, 'smiles')
    with pytest.raises(ValueError):
        m.hydrogens('addHs')

def test_check_rdkit_molecule_exception():
    m = Molecule()
    with pytest.raises(ValueError):
        m.hydrogens('addHs')

def test_mol2xyz_exception(caffeine_smiles):
    m = Molecule()
    m.load(caffeine_smiles, 'smiles')
    m.hydrogens('add')
    with pytest.raises(ValueError):
        m.to_xyz()
    with pytest.raises(ValueError):
        m.to_xyz('uff')

def test_mol2xyz_MMFF(caffeine_smiles):
    m = Molecule()
    m.load(caffeine_smiles, 'smiles')
    m.hydrogens('add')
    m.to_xyz('MMFF', maxIters=210)
    assert m.xyz.geometry.shape[0] == m.rdkit_molecule.GetNumAtoms()
    m.to_xyz('MMFF', maxIters=110, mmffVariant='MMFF94s')
    assert m.xyz.atomic_symbols.shape[0] == m.rdkit_molecule.GetNumAtoms()
    assert m.MMFF_args['mmffVariant'] == 'MMFF94s'
    # check default args
    m.to_xyz('MMFF', **m._default_MMFF_args)

def test_mol2xyz_UFF(caffeine_smiles):
    m = Molecule()
    m.load(caffeine_smiles, 'smiles')
    m.hydrogens('add')
    m.to_xyz('UFF', ignoreInterfragInteractions=False)
    assert m.xyz.geometry.shape[0] == m.rdkit_molecule.GetNumAtoms()
    m.to_xyz('UFF', maxIters=110)
    assert m.xyz.atomic_symbols.shape[0] == m.rdkit_molecule.GetNumAtoms()
    assert m.UFF_args['maxIters'] == 110
    # check default args
    m.to_xyz('UFF', **m._default_UFF_args)
    print (type(m.xyz))