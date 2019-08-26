import pytest

from chemml.chem import bond_features
from chemml.chem import num_bond_features
from chemml.chem import Molecule


@pytest.fixture()
def mols():
    m = Molecule('c1ccc1', 'smiles')
    return m


def test_exception(mols):
    # not an atom
    with pytest.raises(ValueError):
        bond_features(mols)


def test_num_atom_features():
    n = num_bond_features()
    assert n == 6


def test_atom_features(mols):
    bond = mols.rdkit_molecule.GetBonds()[0]
    x = bond_features(bond)

    # double bond
    assert x[1] == 1

    # it's conjugated
    assert x[-1] == 1

    # it's in a ring
    assert x[-2] == 1
