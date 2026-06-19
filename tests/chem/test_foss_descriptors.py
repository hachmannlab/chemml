import sys
import types

import numpy as np
import pandas as pd
import pytest
from rdkit.Chem import Descriptors

from chemml.chem import Molecule, Mordred, PadelDesc, RDKDesc


@pytest.fixture()
def smiles_list():
    return ["CC", "CCO", "CCN"]


@pytest.fixture()
def molecule_list(smiles_list):
    return [Molecule(smi, "smiles") for smi in smiles_list]

@pytest.fixture()
def mixed_inp_list():
    return ["CC", Molecule("CCO", "smiles"), "CCN"]


def test_rdkdesc_represent(monkeypatch, smiles_list, molecule_list):
    desc = RDKDesc()

    # Exercise the joblib branch with real RDKit descriptors.
    desc.descriptor_list = ["MolWt", "HeavyAtomCount"]
    df_parallel = desc.represent(smiles_list, dropna=False, remove_corr=False, n_jobs=2)
    assert df_parallel.shape[0] == len(smiles_list)
    assert set(["MolWt", "HeavyAtomCount", "SMILES"]).issubset(df_parallel.columns)

    # Exercise cleanup logic deterministically with controlled descriptor functions.
    def _const(_mol):
        return 1.0

    def _double_heavy(mol):
        return 2.0 * mol.GetNumHeavyAtoms()

    def _heavy(mol):
        return float(mol.GetNumHeavyAtoms())

    def _with_nan(mol):
        if mol.GetNumHeavyAtoms() == 3:
            return np.nan
        return 0.0

    monkeypatch.setattr(Descriptors, "_test_const", _const, raising=False)
    monkeypatch.setattr(Descriptors, "_test_double_heavy", _double_heavy, raising=False)
    monkeypatch.setattr(Descriptors, "_test_heavy", _heavy, raising=False)
    monkeypatch.setattr(Descriptors, "_test_with_nan", _with_nan, raising=False)

    desc.descriptor_list = ["_test_const", "_test_double_heavy", "_test_heavy", "_test_with_nan"]
    df_dropna = desc.represent(smiles_list, dropna=True, remove_corr=False, n_jobs=1)
    assert "_test_with_nan" not in df_dropna.columns
    assert "SMILES" in df_dropna.columns

    # With <=100 molecules, remove_corr issues a UserWarning and skips removal.
    with pytest.warns(UserWarning):
        df_warn = desc.represent(smiles_list, dropna=True, remove_corr=True, n_jobs=1)
    assert "SMILES" in df_warn.columns
    assert "_test_with_nan" not in df_warn.columns

    # Exercise the Molecule-object input branch.
    df_molecule = desc.represent(molecule_list, dropna=False, remove_corr=False, n_jobs=1)
    assert df_molecule.shape[0] == len(molecule_list)
    assert list(df_molecule["SMILES"]) == [m.smiles for m in molecule_list]

    # Exercise single-string input branch.
    df_single = desc.represent("CC", dropna=False, remove_corr=False, n_jobs=1)
    assert df_single.shape[0] == 1
    assert df_single.loc[0, "SMILES"] == "CC"


def test_mordred_represent(smiles_list, molecule_list):

    class FakeCalcWithMissing:
        def pandas(self, mols, quiet=True):
            return pd.DataFrame(
                {
                    "a": [1.0, np.nan, 3.0],
                    "b": [np.inf, np.nan, np.nan],
                    "c": [7.0, np.inf, 9.0],
                }
            )

    class FakeCalcCorrelated:
        def pandas(self, mols, quiet=True):
            return pd.DataFrame(
                {
                    "x": [1.0, 2.0, 3.0],
                    "y": [2.0, 4.0, 6.0],
                    "z": [1.0, 1.5, 0.5],
                }
            )

    mord = Mordred()

    # Mixed input types should raise an error, as the user should be expected to provide consistent input.
    with pytest.raises(ValueError):
        mord.represent(mixed_inp_list)

    mord.calc = FakeCalcWithMissing()
    df_missing = mord.represent(smiles_list)
    # Modified dropna behaviour drops only columns with all NaN values, not rows; row imputation is left to the user.
    assert list(df_missing.columns) == ["a","c","SMILES"]

    # Exercise Molecule-object input + correlated-feature removal with <=100 molecules.
    # With a small dataset, remove_corr warns and skips correlation removal.
    mord.calc = FakeCalcCorrelated()
    with pytest.warns(UserWarning):
        df_corr = mord.represent(molecule_list, remove_corr=True)
    assert {"x", "y", "z", "SMILES"} <= set(df_corr.columns)

    mord = Mordred(ignore_3D=False)

    with pytest.raises(ValueError):
        mord.represent(smiles_list, optimizer='BADOPT')



def test_padeldesc_represent(smiles_list, molecule_list):
    def fake_from_smiles(smi_list):
        return [
            {"p1": 1.0, "p2": 2.0, "p3": 0.0},
            {"p1": 2.0, "p2": 4.0, "p3": np.nan},
            {"p1": 3.0, "p2": 6.0, "p3": 1.0},
        ][: len(smi_list)]

    fake_module = types.ModuleType("padelpy")
    fake_module.from_smiles = fake_from_smiles
    sys.modules["padelpy"] = fake_module

    padel = PadelDesc()

    # Exercise list-of-smiles input + dropna columns.
    df_smiles = padel.represent(smiles_list, dropna=True, remove_corr=False)
    assert "p3" not in df_smiles.columns
    assert "SMILES" in df_smiles.columns

    # remove_corr drops correlated columns; after dropna removes p3, p1 and p2 are
    # perfectly correlated (r=1.0 > 0.95), so p2 is dropped.
    df_corr = padel.represent(smiles_list, dropna=True, remove_corr=True)
    assert "p1" in df_corr.columns
    assert "p2" not in df_corr.columns
    assert "SMILES" in df_corr.columns

    # Exercise single-Molecule input branch.
    df_single = padel.represent(molecule_list[0], dropna=False, remove_corr=False)
    assert df_single.shape[0] == 1
    assert "SMILES" in df_single.columns
