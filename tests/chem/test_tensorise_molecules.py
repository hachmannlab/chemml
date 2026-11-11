import pytest

from chemml.chem import tensorise_molecules
from chemml.chem import Molecule


@pytest.fixture()
def mols():
    m1 = Molecule('c1ccc1', 'smiles')
    m2 = Molecule('CNC', 'smiles')
    m3 = Molecule('CCc1[nH]ccc1C(=O)N1CC(C2CN(C(=O)/C=C/S(F)(F)(F)(F)F)C2)C1', 'smiles')

    molecules = [m1, m2, m3]

    return molecules


@pytest.mark.parametrize(
    "input_value, expected_exception",
    [
        ("mol", ValueError),
        (["mol1", "mol2"], Exception),
    ],
)
def test_tensorise_molecules_invalid_inputs(input_value, expected_exception):
    with pytest.raises(expected_exception):
        tensorise_molecules(input_value)


def test_tensorise_molecules(mols):
    a, b, d = tensorise_molecules(mols, batch_size=1)

    assert a.shape[0] == 3
    assert b.shape[1] == mols[2].rdkit_molecule.GetNumAtoms()
    assert d.shape[2] == 7
    sulfur = next(atom.GetIdx() for atom in mols[2].rdkit_molecule.GetAtoms()
                  if atom.GetSymbol() == 'S')
    assert (d[2, sulfur] >= 0).sum() == 6

