import pytest

from chemml.chem import tensorise_molecules
from chemml.chem import Molecule


@pytest.fixture()
def mols():
    m1 = Molecule('c1ccc1', 'smiles')
    m2 = Molecule('CNC', 'smiles')

    molecules = [m1, m2]

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

    assert a.shape[0] == 2
    assert b.shape[1] == 4
    assert d.shape[2] == 5

