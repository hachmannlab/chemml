import pytest
import numpy as np

from chemml.chem import BagofBonds
from chemml.chem import Molecule
from chemml.chem import XYZ

@pytest.fixture()
def mols():
    # Oxygen, Hydrogen, Hydrogen
    num = np.array([8, 1, 1])
    num = num.reshape((3, 1))
    sym = np.array(['O', 'H', 'H'])
    sym = sym.reshape((3, 1))
    c = np.array([[1.464, 0.707, 1.056], [0.878, 1.218, 0.498], [2.319, 1.126, 0.952]])
    xyz = XYZ(c,num,sym)
    m = Molecule('O', 'smiles')
    # forcefully overwrite xyz
    m._xyz = xyz
    return m


@pytest.fixture()
def mols2():
    m1 = Molecule('c1ccc1', 'smiles')
    m2 = Molecule('CNC', 'smiles')
    m3 = Molecule('CC', 'smiles')
    m4 = Molecule('CCC', 'smiles')

    molecules = [m1, m2, m3, m4]

    for mol in molecules:
        mol.to_xyz(optimizer='UFF')

    return molecules


def test_h2o(mols):
    bob = BagofBonds(const=1.0)
    h2o_df = bob.represent(mols)

    assert h2o_df.shape == (1, 6)
    a = np.array([[0.66066557, 0.5, 0.5, 8.3593106, 8.35237809, 73.51669472]])

    # O
    ind = bob.header_.index((8.0,))
    assert a[0][5] == pytest.approx(h2o_df.values[0][ind])

    # OH
    ind = bob.header_.index((8.0,1.0))
    assert a[0][3] == pytest.approx(h2o_df.values[0][ind])

    # HH
    ind = bob.header_.index((1.0,1.0))
    assert a[0][0] == pytest.approx(h2o_df.values[0][ind])

    # H
    ind = bob.header_.index((1.0,))
    assert a[0][1] == pytest.approx(h2o_df.values[0][ind])


def test_mollist(mols2):
    bob = BagofBonds(const=1.0, n_jobs=2, verbose=False)
    features = bob.represent(mols2)

    assert features.shape == (4, 13)
    assert bob.header_.count((6.0,6.0)) == 6


def test_exception(mols):
    # Value error: molecule object
    bob = BagofBonds(n_jobs=10)
    with pytest.raises(ValueError):
        bob.represent('fake')
    with pytest.raises(ValueError):
        bob.represent(['fake'])
    # Value error ndim>1
    with pytest.raises(ValueError):
        bob.represent(np.array([[mols],[mols]]))
