import pytest
import os
from json import dump
import tempfile
import shutil
import warnings

from chemml.models.keras import MLP
from chemml.datasets import load_organic_density


@pytest.fixture()
def data():
    _, y, X = load_organic_density()
    y = y.values.reshape(y.shape[0], 1).astype('float32')
    X = X.values.reshape(X.shape[0], X.shape[1]).astype('float32')

    # scale roughly
    y = y / 1000.0
    X = X / 10.0

    # split 0.9 train / 0.1 test
    ytr = y[:450, :]
    yte = y[450:, :]
    Xtr = X[:450, :]
    Xte = X[450:, :]
    return Xtr, ytr, Xte, yte


@pytest.fixture()
def setup_teardown():
    # Create a temporary directory
    test_dir = tempfile.mkdtemp()
    # return test directory to save figures
    yield test_dir
    # Remove the directory after the test
    shutil.rmtree(test_dir)


def test_init_via_config(data, setup_teardown):
    # make layer config file
    layer_config = [('Dense', {
        'units': 64,
        'activation': 'relu'
    }), ('Dropout', {
        'rate': 0.2
    }), ('Dense', {
        'units': 32,
        'activation': 'relu'
    }), ('Dropout', {
        'rate': 0.2
    }), ('Dense', {
        'units': 1,
        'activation': 'linear'
    })]
    temp_path = setup_teardown
    with open(os.path.join(temp_path, 'layers.config'), 'w') as f:
        dump(layer_config, f)

    # make opt config file
    opt = ['SGD', {'lr': 0.1, 'momentum': 0.9}]
    with open(os.path.join(temp_path, 'opt.config'), 'w') as f:
        dump(opt, f)

    mlp = MLP(
        regression=True,
        nepochs=5,
        batch_size=20,
        loss='mse',
        layer_config_file=os.path.join(temp_path, 'layers.config'),
        opt_config_file=os.path.join(temp_path, 'opt.config'))
    Xtr, ytr, Xte, yte = data
    # mlp.model.summary()
    warnings.simplefilter("always")
    with warnings.catch_warnings(record=True) as w:
        mlp.fit(Xtr, ytr)
        mlp.score(Xte, yte)


def test_init_via_params(data):
    Xtr, ytr, Xte, yte = data
    mlp = MLP(
        nhidden=2,
        nneurons=[64, 32],
        activations=['relu', 'relu'],
        learning_rate=0.1,
        nepochs=5,
        batch_size=20,
        loss='mse',
        regression=True)
    # mlp.model.summary()
    warnings.simplefilter("always")
    with warnings.catch_warnings(record=True) as w:
        mlp.fit(Xtr, ytr)
        mlp.score(Xte, yte)
