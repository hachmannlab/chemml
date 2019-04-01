from chemml.datasets import load_cep_homo
from chemml.datasets import load_organic_density
from chemml.datasets import load_xyz_polarizability
from chemml.datasets import load_comp_energy
from chemml.datasets import load_crystal_structures
from chemml.chem import Molecule

def test_load_cep_homo():
    smi, homo = load_cep_homo()
    assert smi.columns[0] == 'smiles'
    assert homo.shape[0] == 500


def test_load_organic_density():
    smi, density, features = load_organic_density()
    assert smi.columns[0] == 'smiles'
    assert density.shape == (500, 1)
    assert features.shape == (500, 200)


def test_load_xyz_polarizability():
    molecules, df = load_xyz_polarizability()
    assert isinstance(molecules, list)
    assert '17_opt.xyz' in molecules[16].creator[1]
    assert '37_opt.xyz' in molecules[36].creator[1]
    assert molecules[7].xyz is not None
    assert df.shape == (50, 1)


def test_load_comp_energy():
    entries, df = load_comp_energy()
    assert len(entries) == 630
    assert df.shape == (630, 1)


def test_load_crystal_structures():
    entries = load_crystal_structures()
    assert len(entries) == 18
