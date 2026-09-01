import pytest
import os
import shutil
import tempfile


from chemml.chem import RDKitFingerprint
from chemml.chem import Molecule


# @pytest.fixture()
# def data_path():
#     return pkg_resources.resource_filename(
#         'chemml', os.path.join('datasets', 'data', 'test_files'))


@pytest.fixture()
def mol_single():
    smi1 = Molecule('c1cc2cnc3c(cnc4cc(-c5ncncn5)c5nsnc5c34)c2c2nsnc12', 'smiles')
    return smi1

@pytest.fixture()
def mol_list():
    smi1 = Molecule('c1cc2cnc3c(cnc4cc(-c5ncncn5)c5nsnc5c34)c2c2nsnc12', 'smiles')
    smi2 = Molecule('[nH]1ccc2[nH]c3c4CC(=Cc4c4c[nH]cc4c3c12)c1scc2cc[nH]c12', 'smiles')
    return [smi1, smi2]

@pytest.fixture()
def setup_teardown():
    # Create a temporary directory
    test_dir = tempfile.mkdtemp()
    # return test directory to save figures
    yield test_dir
    # Remove the directory after the test
    shutil.rmtree(test_dir)

def test_vector_exception():
    with pytest.raises(ValueError):
        _ = RDKitFingerprint(vector='Integer')


def test_molecules_exception(mol_list):
    with pytest.raises(ValueError):
        rdfp = RDKitFingerprint()
        rdfp.represent(tuple(mol_list))
    with pytest.raises(ValueError):
        rdfp = RDKitFingerprint()
        rdfp.represent('fake molecule')


def test_type_exception(mol_single):
    with pytest.raises(ValueError):
        rdfp = RDKitFingerprint(fingerprint_type='fake')
        rdfp.represent(mol_single)


@pytest.mark.parametrize(
    "fingerprint_type, vector, kwargs, expected_shapes",
    [
        ("hap", "int", {}, ((2, 107), (1, 90))),
        ("hap", "bit", {}, ((2, 1024), (1, 1024))),
        ("maccs", "bit", {}, ((2, 167), (1, 167))),
        ("Morgan", "int", {}, ((2, 84), (1, 44))),
        ("Morgan", "bit", {"radius": 3, "useChirality": True, "useBondTypes": True, "useFeatures": True}, ((2, 1024), (1, 1024))),
        ("htt", "int", {}, ((2, 41), (1, 22))),
        ("htt", "bit", {}, ((2, 1024), (1, 1024))),
        ("tt", "int", {}, ((2, 42), (1, 22))),
        ("tt", "bit", {}, ((2, 1024), (1, 1024))),
    ],
)
def test_fingerprint_representations(mol_list, mol_single, fingerprint_type, vector, kwargs, expected_shapes):
    rdfp = RDKitFingerprint(fingerprint_type=fingerprint_type, vector=vector, **kwargs)
    df = rdfp.represent(mol_list)
    assert df.shape == expected_shapes[0]
    assert rdfp.n_molecules_ == 2
    df = rdfp.represent(mol_single)
    assert df.shape == expected_shapes[1]
    assert rdfp.n_molecules_ == 1


def test_MACCS_exception(mol_list):
    with pytest.raises(ValueError):
        rdfp = RDKitFingerprint(fingerprint_type='maccs', vector='int')
        rdfp.represent(mol_list)

def test_store_sparse(mol_list, setup_teardown):
    rdfp = RDKitFingerprint(fingerprint_type='morgan', vector='bit')
    df = rdfp.represent(mol_list)
    temp_file = os.path.join(setup_teardown, 'temp.nzp')
    rdfp.store_sparse(temp_file, df)

