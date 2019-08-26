import pytest

from chemml.chem import tensorise_molecules
from chemml.chem import Molecule


@pytest.fixture()
def mols():
    m1 = Molecule('c1ccc1', 'smiles')
    m2 = Molecule('CNC', 'smiles')

    molecules = [m1, m2]

    return molecules


def test_exception():
    # not a molecule
    with pytest.raises(ValueError):
        tensorise_molecules('mol')

    # not a list of molecules
    with pytest.raises(Exception):
        tensorise_molecules(['mol1', 'mol2'])


def test_tensorise_molecules(mols):

    a,b,d = tensorise_molecules(mols, batch_size=1)

    assert a.shape[0] == 2
    assert b.shape[1] == 4
    assert d.shape[2] == 5

