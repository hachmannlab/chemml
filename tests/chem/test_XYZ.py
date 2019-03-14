import pytest
import numpy as np

from chemml.chem import XYZ


@pytest.fixture()
def caffeine_smiles():
    smiles = 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'
    return smiles


def test_instantiate():
    g = np.array([[ 3.09002369e+00,  1.41663512e+00, -6.09700287e-02],
                  [ 2.19236533e+00,  2.88035321e-01, -3.67414204e-02],
                  [ 2.57936056e+00, -1.02327782e+00, -3.08471940e-02]])
    n = np.array([[6], [1], [8]])
    s = np.array([['C'], ['H'], ['O']])
    m = XYZ(g, n, s)

def test_exception():
    g = np.array([[ 3.09002369e+00,  1.41663512e+00, -6.09700287e-02],
                  [ 2.19236533e+00,  2.88035321e-01, -3.67414204e-02],
                  [ 2.57936056e+00, -1.02327782e+00, -3.08471940e-02]])
    n = [[6], [1], [8]]
    s = np.array([['C'], ['H'], ['O']])
    # numpy
    with pytest.raises(ValueError):
        m = XYZ(g, n, s)
    # cast
    n = np.array([['C'], [1], [8]])
    with pytest.raises(ValueError):
        m = XYZ(g, n, s)
    # shape: geometry
    g = np.array([[ 3.09002369e+00,  1.41663512e+00, -6.09700287e-02],
                  [ 2.19236533e+00,  2.88035321e-01, -3.67414204e-02]])
    n = np.array([[6], [1], [8]])
    with pytest.raises(ValueError):
        m = XYZ(g, n, s)
    # shape: all
    g = np.array([[ 3.09002369e+00,  1.41663512e+00, -6.09700287e-02],
                  [ 2.19236533e+00,  2.88035321e-01, -3.67414204e-02],
                  [ 2.57936056e+00, -1.02327782e+00, -3.08471940e-02]])
    n = np.array([[6], [1]])
    with pytest.raises(ValueError):
        m = XYZ(g, n, s)
