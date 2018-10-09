import pytest
import os
import pkg_resources

from chemml.chem import RDKitFingerprint


@pytest.fixture()
def data_path():
    return pkg_resources.resource_filename(
        'chemml', os.path.join('datasets', 'data', 'test_files'))


def test_instantiate(data_path):
    cls = RDKitFingerprint()
    cls.read_(os.path.join(data_path, 'smiles.smi'))
    df = cls.fingerprint()


def test_parser(data_path):
    cls = RDKitFingerprint(fingerprint_type='hap')
    cls.read_('smiles/mol_[1-3].smi', data_path)
    assert len(cls.molecules) == 2
    # check number of bits
    df = cls.fingerprint()
    assert df.shape[1] == 1024


def test_extension_exception(data_path):
    with pytest.raises(ValueError):
        cls = RDKitFingerprint()
        cls.read_(os.path.join(data_path, 'Dragon_script.drs'))


def test_vector_exception(data_path):
    with pytest.raises(ValueError):
        cls = RDKitFingerprint(vector='binary')
        cls.read_(os.path.join(data_path, 'smiles.smi'))


def test_type_exception(data_path):
    with pytest.raises(ValueError):
        cls = RDKitFingerprint(fingerprint_type='HP')
        cls.read_(os.path.join(data_path, 'smiles.smi'))
        cls.fingerprint()


def test_hap_int(data_path):
    cls = RDKitFingerprint(fingerprint_type='hap',vector='int')
    cls.read_('smiles/mol_[1-3].smi', data_path)
    df = cls.fingerprint()
    assert df.shape[1] == 107


def test_MACCS_bit(data_path):
    cls = RDKitFingerprint(fingerprint_type='MACCS', vector='bit')
    cls.read_('smiles/mol_[1-3].smi', data_path)
    df = cls.fingerprint()
    assert df.shape[1] == 167


def test_MACCS_int_exception(data_path):
    with pytest.raises(ValueError):
        cls = RDKitFingerprint(fingerprint_type='MACCS', vector='int')
        cls.read_('smiles/mol_[1-3].smi', data_path)
        cls.fingerprint()


def test_Morgan_int(data_path):
    cls = RDKitFingerprint(fingerprint_type='Morgan', vector='int')
    cls.read_('smiles/mol_[1-3].smi', data_path)
    df = cls.fingerprint()
    assert df.shape[1] == 84


def test_Morgan_bit(data_path):
    cls = RDKitFingerprint(fingerprint_type='Morgan', vector='bit')
    cls.read_('smiles/mol_[1-3].smi', data_path)
    df = cls.fingerprint()
    assert df.shape[1] == 1024


def test_htt_int(data_path):
    cls = RDKitFingerprint(fingerprint_type='htt', vector='int')
    cls.read_('smiles/mol_[1-3].smi', data_path)
    df = cls.fingerprint()
    assert df.shape[1] == 41


def test_htt_bit(data_path):
    cls = RDKitFingerprint(fingerprint_type='htt', vector='bit')
    cls.read_('smiles/mol_[1-3].smi', data_path)
    df = cls.fingerprint()
    assert df.shape[1] == 1024


def test_tt_int(data_path):
    cls = RDKitFingerprint(fingerprint_type='tt', vector='int')
    cls.read_('smiles/mol_[1-3].smi', data_path)
    df = cls.fingerprint()
    assert df.shape[1] == 42

def test_tt_bit_exception(data_path):
    with pytest.raises(ValueError):
        cls = RDKitFingerprint(fingerprint_type='tt', vector='bit')
        cls.read_('smiles/mol_[1-3].smi', data_path)
        cls.fingerprint()
