import pytest
from chemml.utils import ConvertFile
import pkg_resources
import os



@pytest.fixture()
def xyz_path():
    return pkg_resources.resource_filename(
        'chemml', os.path.join('datasets', 'data', 'organic_xyz'))



def test_convertfile(xyz_path):
    
    cf = ConvertFile(xyz_path + '/1_opt.xyz', 'xyz', 'mol2')
    assert cf.split('.')[-1] == 'mol2'
    
    cf = ConvertFile([xyz_path + '/1_opt.xyz', xyz_path + '/2_opt.xyz'], 'xyz', 'sdf')
    assert cf[0].split('.')[-1] == 'sdf'
    assert cf[1].split('.')[-1] == 'sdf'
    
    with pytest.raises(ValueError):
        cf = ConvertFile({'a.sdf':1}, 'xyz', 'mol2')
    
