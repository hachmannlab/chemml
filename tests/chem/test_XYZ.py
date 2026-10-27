import pytest
import numpy as np

from chemml.chem import XYZ


@pytest.fixture()
def caffeine_smiles():
    smiles = 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'
    return smiles


def test_instantiate():
    g = np.array([[3.09002369e+00, 1.41663512e+00, -6.09700287e-02],
                  [2.19236533e+00, 2.88035321e-01, -3.67414204e-02],
                  [2.57936056e+00, -1.02327782e+00, -3.08471940e-02]])
    n = np.array([[6], [1], [8]])
    s = np.array([['C'], ['H'], ['O']])
    m = XYZ(g, n, s)


@pytest.mark.parametrize(
    "geometry, atomic_numbers, atomic_symbols",
    [
        (np.array([[3.09002369e+00, 1.41663512e+00, -6.09700287e-02],
                   [2.19236533e+00, 2.88035321e-01, -3.67414204e-02],
                   [2.57936056e+00, -1.02327782e+00, -3.08471940e-02]]),
         [[6], [1], [8]],
         np.array([['C'], ['H'], ['O']])),
        (np.array([[3.09002369e+00, 1.41663512e+00, -6.09700287e-02],
                   [2.19236533e+00, 2.88035321e-01, -3.67414204e-02],
                   [2.57936056e+00, -1.02327782e+00, -3.08471940e-02]]),
         np.array([['C'], [1], [8]]),
         np.array([['C'], ['H'], ['O']])),
        (np.array([[3.09002369e+00, 1.41663512e+00, -6.09700287e-02],
                   [2.19236533e+00, 2.88035321e-01, -3.67414204e-02]]),
         np.array([[6], [1], [8]]),
         np.array([['C'], ['H'], ['O']])),
        (np.array([[3.09002369e+00, 1.41663512e+00, -6.09700287e-02],
                   [2.19236533e+00, 2.88035321e-01, -3.67414204e-02],
                   [2.57936056e+00, -1.02327782e+00, -3.08471940e-02]]),
         np.array([[6], [1]]),
         np.array([['C'], ['H'], ['O']])),
    ],
)
def test_exception(geometry, atomic_numbers, atomic_symbols):
    with pytest.raises(ValueError):
        XYZ(geometry, atomic_numbers, atomic_symbols)
