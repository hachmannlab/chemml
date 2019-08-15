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


def test_hap_int(mol_list, mol_single):
    rdfp = RDKitFingerprint(fingerprint_type='hap',vector='int')
    df = rdfp.represent(mol_list)
    assert df.shape == (2, 107)
    assert rdfp.n_molecules_ == 2
    df = rdfp.represent(mol_single)
    assert df.shape == (1, 90)
    assert rdfp.n_molecules_ == 1


def test_hap_bit(mol_list, mol_single):
    rdfp = RDKitFingerprint(fingerprint_type='hap',vector='bit')
    df = rdfp.represent(mol_list)
    assert df.shape == (2, 1024)
    assert rdfp.n_molecules_ == 2
    df = rdfp.represent(mol_single)
    assert df.shape == (1, 1024)
    assert rdfp.n_molecules_ == 1


def test_MACCS_exception(mol_list, mol_single):
    with pytest.raises(ValueError):
        rdfp = RDKitFingerprint(fingerprint_type='maccs', vector='int')
        df = rdfp.represent(mol_list)


def test_MACCS(mol_list, mol_single):
    rdfp = RDKitFingerprint(fingerprint_type='maccs', vector='bit')
    df = rdfp.represent(mol_list)
    assert df.shape == (2, 167)
    assert rdfp.n_molecules_ == 2
    df = rdfp.represent(mol_single)
    assert df.shape == (1, 167)
    assert rdfp.n_molecules_ == 1


def test_Morgan_int(mol_list, mol_single):
    rdfp = RDKitFingerprint(fingerprint_type='Morgan', vector='int')
    df = rdfp.represent(mol_list)
    assert df.shape == (2, 84)
    assert rdfp.n_molecules_ == 2
    df = rdfp.represent(mol_single)
    assert df.shape == (1, 44)
    assert rdfp.n_molecules_ == 1


def test_Morgan_bit(mol_list, mol_single):
    rdfp = RDKitFingerprint(fingerprint_type='Morgan', vector='bit')
    df = rdfp.represent(mol_list)
    assert df.shape == (2, 1024)
    assert rdfp.n_molecules_ == 2
    df = rdfp.represent(mol_single)
    assert df.shape == (1, 1024)
    assert rdfp.n_molecules_ == 1
    # kwargs
    rdfp = RDKitFingerprint(fingerprint_type='Morgan', vector='bit', radius = 3,
                            useChirality=True, useBondTypes=True, useFeatures=True)
    df = rdfp.represent(mol_list)
    assert df.shape == (2, 1024)
    df = rdfp.represent(mol_single)
    assert df.shape == (1, 1024)
    assert rdfp.n_molecules_ == 1


def test_htt_int(mol_list, mol_single):
    rdfp = RDKitFingerprint(fingerprint_type='htt', vector='int')
    df = rdfp.represent(mol_list)
    assert df.shape == (2, 41)
    assert rdfp.n_molecules_ == 2
    df = rdfp.represent(mol_single)
    assert df.shape == (1, 22)
    assert rdfp.n_molecules_ == 1


def test_htt_bit(mol_list, mol_single):
    rdfp = RDKitFingerprint(fingerprint_type='htt', vector='bit')
    df = rdfp.represent(mol_list)
    assert df.shape == (2, 1024)
    assert rdfp.n_molecules_ == 2
    df = rdfp.represent(mol_single)
    assert df.shape == (1, 1024)
    assert rdfp.n_molecules_ == 1


def test_tt_int(mol_list, mol_single):
    rdfp = RDKitFingerprint(fingerprint_type='tt', vector='int')
    df = rdfp.represent(mol_list)
    assert df.shape == (2, 42)
    assert rdfp.n_molecules_ == 2
    df = rdfp.represent(mol_single)
    assert df.shape == (1, 22)
    assert rdfp.n_molecules_ == 1


def test_tt_bit_exception(mol_single):
    with pytest.raises(ValueError):
        cls = RDKitFingerprint(fingerprint_type='tt', vector='bit')
        cls.represent(mol_single)

def test_store_sparse(mol_list, setup_teardown):
    rdfp = RDKitFingerprint(fingerprint_type='morgan', vector='bit')
    df = rdfp.represent(mol_list)
    temp_file = os.path.join(setup_teardown, 'temp.nzp')
    rdfp.store_sparse(temp_file, df)

