import pytest
import shutil
import tempfile

from chemml.chem import Dragon, Molecule


@pytest.fixture()
def setup_teardown():
    # Create a temporary directory
    test_dir = tempfile.mkdtemp()
    # return test directory to save figures
    yield test_dir
    # Remove the directory after the test
    shutil.rmtree(test_dir)


@pytest.fixture()
def chemml_molecule_list():
    molecules = ['C', 'CC', 'CCC', 'CC(C)C']
    mol_list = [Molecule(i, 'smiles') for i in molecules]
    yield mol_list


@pytest.mark.parametrize(
    "kwargs, expected_error",
    [
        ({"Weights": ["a"]}, ValueError),
        ({"blocks": list(range(2, 32))}, ValueError),
    ],
)
def test_dragon_invalid_inputs_raise(setup_teardown, chemml_molecule_list, kwargs, expected_error):
    try:
        drg = Dragon(**kwargs)
        with pytest.raises(expected_error):
            drg.represent(mol_list=chemml_molecule_list, output_directory=setup_teardown)
    except ImportError:
        pytest.skip("Dragon requires external dependencies")


@pytest.mark.parametrize(
    "mol_list, expected_error",
    [
        ([], ValueError),
        ("CC", ValueError),
        (["CC"], ValueError),
    ],
)
def test_dragon_invalid_mol_list_inputs(setup_teardown, mol_list, expected_error):
    try:
        drg = Dragon()
        with pytest.raises(expected_error):
            drg.represent(mol_list=mol_list, output_directory=setup_teardown, dropna=False)
    except ImportError:
        pytest.skip("Dragon requires external dependencies")


def test_dragon_df(setup_teardown, chemml_molecule_list):
    try:
        drg = Dragon()
        df = drg.represent(mol_list=chemml_molecule_list, output_directory=setup_teardown)
        assert df.shape[0] == 4
    except ImportError:
        pytest.skip("Dragon requires external dependencies")
