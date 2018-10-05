import pytest
import os
import shutil
import tempfile
import pkg_resources
import warnings

from chemml.chem import Dragon

@pytest.fixture()
def data_path():
    return pkg_resources.resource_filename('chemml', os.path.join('datasets', 'data', 'test_files'))

@pytest.fixture()
def setup_teardown():
    # Create a temporary directory
    test_dir = tempfile.mkdtemp()
    # return test directory to save figures
    yield test_dir
    # Remove the directory after the test
    shutil.rmtree(test_dir)

def test_new_script_v8():
    with pytest.raises(ValueError):
        drg = Dragon(version=8)


def test_existing_script(data_path, setup_teardown):
    drg = Dragon()
    drg.script_wizard(script=os.path.join(data_path, 'Dragon_script.drs'), output_directory=setup_teardown)


def test_existing_script_exception(data_path, setup_teardown):
    with pytest.warns(RuntimeWarning):
        with pytest.raises(ValueError):
            drg = Dragon()
            drg.script_wizard(script=os.path.join(data_path, 'Dragon_script_broken.drs'), output_directory=setup_teardown)


def test_new_script_v7(setup_teardown):
    # test script exists
    drg = Dragon()
    drg.script_wizard(script='new', output_directory=setup_teardown)
    assert os.path.exists(os.path.join(drg.output_directory, 'Dragon_script.drs')) == 1
    drg.printout()


def test_v7_weights_exception(setup_teardown):
    with pytest.raises(ValueError):
        drg = Dragon(Weights=['a'])
        drg.script_wizard(script='new', output_directory=setup_teardown)


def test_v7_blocks_exception(setup_teardown):
    with pytest.raises(ValueError):
        drg = Dragon(blocks=list(range(2, 32)))
        drg.script_wizard(script='new', output_directory=setup_teardown)


def test_v7_molinput_exception(setup_teardown):
    with pytest.raises(ValueError):
        drg = Dragon(molInput='stdin')
        drg.script_wizard(script='new', output_directory=setup_teardown)


def test_v7_molFile_exception(data_path, setup_teardown):
    with pytest.raises(ValueError):
        drg = Dragon(molFile=[])
        drg.script_wizard(script='new', output_directory=setup_teardown)


def test_new_script_v6(setup_teardown):
    # test script exists
    drg = Dragon(version=6, blocks=list(range(1,30)))
    drg.script_wizard(script='new', output_directory=setup_teardown)
    assert os.path.exists(os.path.join(drg.output_directory, 'Dragon_script.drs')) == 1
    # drg.printout()


def test_v6_weights_exception(setup_teardown):
    with pytest.raises(ValueError):
        drg = Dragon(version=6, blocks=list(range(1,30)), Weights=['a'])
        drg.script_wizard(script='new', output_directory=setup_teardown)


def test_v6_blocks_exception(setup_teardown):
    with pytest.raises(ValueError):
        drg = Dragon(version=6, blocks=list(range(2, 32)))
        drg.script_wizard(script='new', output_directory=setup_teardown)


def test_v6_molinput_exception(setup_teardown):
    with pytest.raises(ValueError):
        drg = Dragon(version=6, blocks=list(range(1,30)), molInput='stdin')
        drg.script_wizard(script='new', output_directory=setup_teardown)


def test_v6_molFile_exception(setup_teardown):
    with pytest.raises(ValueError):
        drg = Dragon(version=6, blocks=list(range(1,30)), molFile=[])
        drg.script_wizard(script='new', output_directory=setup_teardown)

def test_run(setup_teardown):
    with pytest.raises(ImportError):
        drg = Dragon(version=6, blocks=list(range(1, 30)))
        drg.script_wizard(script='new', output_directory=setup_teardown)
        drg.run()
