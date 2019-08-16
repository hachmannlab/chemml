import pytest

from chemml.chem import atom_features
from chemml.chem import num_atom_features
from chemml.chem import Molecule


@pytest.fixture()
def mols():
    m = Molecule('CCO', 'smiles')
    return m


def test_exception(mols):
    # not an atom
    with pytest.raises(ValueError):
        atom_features(mols)


def test_num_atom_features(mols):
    n = num_atom_features()
    assert n == 62


def test_atom_features(mols):
    atom = mols.rdkit_molecule.GetAtoms()[0]
    x = atom_features(atom)

    # it's a carbon
    assert x[0] == 1

    # it's not aromatic
    assert x[-1] == 0

    # it's implicit valence is 3
    assert x[-4] == 1
