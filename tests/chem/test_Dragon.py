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


def test_weights_exception(setup_teardown, chemml_molecule_list):
    with pytest.raises(ValueError):
        drg = Dragon(Weights=['a'])
        df = drg.represent(mol_list=chemml_molecule_list, output_directory=setup_teardown)


def test_blocks_exception(setup_teardown, chemml_molecule_list):
    with pytest.raises(ValueError):
        drg = Dragon(blocks=list(range(2, 32)))
        df = drg.represent(mol_list=chemml_molecule_list, output_directory=setup_teardown)


def test_empty_mol_list(setup_teardown):
    with pytest.raises(ValueError):
        drg = Dragon()
        # print(drg, drg.getattr())
        df = drg.represent(mol_list=[], output_directory=setup_teardown, dropna=False)



def test_input_str(setup_teardown):
    with pytest.raises(ValueError):
        drg = Dragon()
        df = drg.represent(mol_list='CC', output_directory=setup_teardown)
        

def test_input_list(setup_teardown):
    with pytest.raises(ValueError):
        drg = Dragon()
        df = drg.represent(mol_list=['CC'], output_directory=setup_teardown)



def test_dragon_df(setup_teardown, chemml_molecule_list):
    drg = Dragon()
    df = drg.represent(mol_list=chemml_molecule_list, output_directory=setup_teardown)
    assert df.shape[0] == 4



