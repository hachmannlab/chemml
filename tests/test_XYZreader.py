import pytest
import os
import pkg_resources

from chemml.initialization import XYZreader


@pytest.fixture()
def data_path():
    return pkg_resources.resource_filename('chemml', os.path.join('datasets', 'data', 'organic_xyz'))


def test_string_manual(data_path):
    reader = XYZreader(
        path_pattern='[2-3]_opt.xyz', path_root=data_path, reader='manual', skip_lines=[2, 0])
    molecules = reader.read()
    assert len(molecules) == 2
    assert len(molecules[1]['mol']) == 20


def test_list_manual(data_path):
    reader = XYZreader(
        path_pattern=['[2-3]_opt.xyz', '[1-2][1-2]_opt.xyz'],
        path_root=data_path,
        reader='manual',
        skip_lines=[2, 0])
    molecules = reader.read()
    assert len(molecules) == 6
    assert len(molecules[1]['mol']) == 20
    assert len(molecules[4]['mol']) == 24
