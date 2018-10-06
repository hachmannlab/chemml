import pytest
import os
import pkg_resources

from chemml.chem import RDKitFingerprint


@pytest.fixture()
def data_path():
    return pkg_resources.resource_filename(
        'chemml', os.path.join('datasets', 'data', 'organic_xyz'))


def test_instantiate(data_path):
    cls = RDKitFingerprint()
