from multiprocessing.dummy.connection import families
import pytest
import os
from json import dump
import tempfile
import shutil
import warnings
import numpy as np

import tensorflow as tf
# tf.get_logger().setLevel(3) #to suppress warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from chemml.models import MLP
from chemml.datasets import load_organic_density
from chemml.utils import regression_metrics
from torch import nn
from torch.nn import Sequential as pytSeq
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential as tfSeq


def test_init():
    # PYTORCH
    r1 = MLP(engine='pytorch',nfeatures=120, nneurons=[100,200], activations=['ReLU','ReLU'],
                learning_rate=0.01, alpha=0.002, nepochs=100, batch_size=100, loss='mean_squared_error', 
                is_regression=True, nclasses=None, layer_config_file=None, opt_config='SGD')
    c1 = MLP(engine='pytorch',nfeatures=120, nneurons=[100,200], activations=['ReLU','Sigmoid'],
                learning_rate=0.01, alpha=0.201, nepochs=100, batch_size=100, loss='mean_squared_error', 
                is_regression=False, nclasses=2, layer_config_file=None, opt_config='ADAM')
                
    assert isinstance(r1, MLP)
    assert isinstance(r1.model, pytSeq)
    
    assert isinstance(c1, MLP)
    assert isinstance(c1.model, pytSeq)

    # TENSORFLOW
    r1 = MLP(engine='tensorflow',nfeatures=120, nneurons=[100,200], activations=['ReLU','ReLU'],
                learning_rate=0.01, alpha=0.002, nepochs=100, batch_size=100, loss='mean_squared_error', 
                is_regression=True, nclasses=None, layer_config_file=None, opt_config='SGD')
    c1 = MLP(engine='tensorflow',nfeatures=120, nneurons=[100,200], activations=['ReLU','Sigmoid'],
                learning_rate=0.01, alpha=0.201, nepochs=100, batch_size=100, loss='mean_squared_error', 
                is_regression=False, nclasses=2, layer_config_file=None, opt_config='ADAM')
    assert isinstance(r1, MLP)
    assert isinstance(r1.model, tfSeq)

    assert isinstance(c1, MLP)
    assert isinstance(c1.model, tfSeq)



@pytest.fixture()
def data():
    _, y, X = load_organic_density()
    y = y.values.reshape(y.shape[0], 1).astype('float32')
    X = X.values.reshape(X.shape[0], X.shape[1]).astype('float32')

    # split 0.9 train / 0.1 test
    ytr = y[:450, :]
    yte = y[450:, :]
    Xtr = X[:450, :]
    Xte = X[450:, :]
    
    scale = StandardScaler()
    scale_y = StandardScaler()
    Xtr = scale.fit_transform(Xtr)
    Xte = scale.transform(Xte)
    ytr = scale_y.fit_transform(ytr)
    return Xtr, ytr, Xte, yte, scale_y


def test_fit_via_params(data):
    Xtr, ytr, Xte, yte, scale_y = data

    mlp_pytorch = MLP(engine='pytorch', nfeatures=Xtr.shape[1], nneurons=[100,200], activations=['ReLU','ReLU'],
                learning_rate=0.01, alpha=0.002, nepochs=20, batch_size=100, loss='mean_squared_error', 
                is_regression=True, nclasses=None, layer_config_file=None, opt_config='SGD'
        )
    mlp_pytorch.fit(Xtr, ytr)
    y_pred = mlp_pytorch.predict(Xte).reshape(-1,1)
    y_pred = scale_y.inverse_transform(y_pred)

    metrics_df = regression_metrics(yte, y_pred)
    assert isinstance(metrics_df['MAE'].loc[0],np.float32)
    

    mlp_tensorflow = MLP(engine='tensorflow', nfeatures=Xtr.shape[1], nneurons=[100,200], activations=['ReLU','ReLU'],
                learning_rate=0.01, alpha=0.002, nepochs=20, batch_size=100, loss='mean_squared_error', 
                is_regression=True, nclasses=None, layer_config_file=None, opt_config='SGD'
        )
    mlp_tensorflow.fit(Xtr, ytr)
    y_pred = mlp_tensorflow.predict(Xte).reshape(-1,1)
    y_pred = scale_y.inverse_transform(y_pred)

    metrics_df = regression_metrics(yte, y_pred)
    assert isinstance(metrics_df['MAE'].loc[0],np.float32)


@pytest.fixture()
def setup_teardown():
    # Create a temporary directory
    test_dir = tempfile.mkdtemp()
    # return test directory to save figures
    yield test_dir
    # Remove the directory after the test
    shutil.rmtree(test_dir)


def test_get_model():
    # TENSORFLOW
    r1_tensorflow = MLP(engine='tensorflow',nfeatures=120, nneurons=[100,200,300], activations=['ReLU','ReLU','ReLU'],
                learning_rate=0.01, alpha=0.002, nepochs=100, batch_size=100, loss='mean_squared_error', 
                is_regression=True, nclasses=None, layer_config_file=None, opt_config='SGD')
    engine_model = r1_tensorflow.get_model()
    assert len(engine_model.layers) == 4


    engine_model_1 = r1_tensorflow.get_model(include_output=False)
    assert len(engine_model_1.layers) == 3
    del r1_tensorflow

    r1_tensorflow = MLP(engine='tensorflow',nfeatures=120, nneurons=[100,200,300], activations=['ReLU','ReLU','ReLU'],
            learning_rate=0.01, alpha=0.002, nepochs=100, batch_size=100, loss='mean_squared_error', 
            is_regression=True, nclasses=None, layer_config_file=None, opt_config='SGD')

    engine_model_2 = r1_tensorflow.get_model(include_output=False,n_layers=1)
    assert len(engine_model_2.layers) == 2

    # PYTORCH
    r1_pytorch = MLP(engine='pytorch',nfeatures=120, nneurons=[100,200,300], activations=['ReLU','ReLU','ReLU'],
            learning_rate=0.01, alpha=0.002, nepochs=100, batch_size=100, loss='mean_squared_error', 
            is_regression=True, nclasses=None, layer_config_file=None, opt_config='SGD')

    engine_model = r1_pytorch.get_model()
    print(engine_model)
    assert len(engine_model) == 7

    engine_model = r1_pytorch.get_model(include_output=False)
    assert len(engine_model) == 7
    print(engine_model)

    r2_pytorch = MLP(engine='pytorch',nfeatures=120, nneurons=[100,200,300], activations=['ReLU','ReLU','ReLU'],
            learning_rate=0.01, alpha=0.002, nepochs=100, batch_size=100, loss='mean_squared_error', 
            is_regression=True, nclasses=None, layer_config_file=None, opt_config='SGD')

    engine_model = r2_pytorch.get_model(n_layers=3)
    print(engine_model)

    assert len(engine_model) == 4


def test_multioutput(data):
    """Test multi-output regression for both PyTorch and TensorFlow engines."""
    Xtr, ytr, Xte, yte, scale_y = data
    
    # Create 2-output targets by duplicating and scaling the original output
    ytr_multi = np.hstack([ytr, ytr * 0.5])  # shape: (450, 2)
    yte_multi = np.hstack([yte, yte * 0.5])  # shape: (50, 2)
    
    # Test PyTorch engine
    mlp_pytorch = MLP(engine='pytorch', nfeatures=Xtr.shape[1], nneurons=[100, 200], 
                      activations=['ReLU', 'ReLU'], learning_rate=0.01, alpha=0.002, 
                      nepochs=20, batch_size=100, loss='mean_squared_error', 
                      is_regression=True, noutputs=2, opt_config='SGD')
    
    assert mlp_pytorch.noutputs == 2
    params = mlp_pytorch.get_params()
    assert params['noutputs'] == 2
    
    mlp_pytorch.fit(Xtr, ytr_multi)
    y_pred_pytorch = mlp_pytorch.predict(Xte)
    assert y_pred_pytorch.shape == (yte_multi.shape[0], 2), f"Expected shape (50, 2), got {y_pred_pytorch.shape}"
    
    # Test TensorFlow engine
    mlp_tensorflow = MLP(engine='tensorflow', nfeatures=Xtr.shape[1], nneurons=[100, 200], 
                        activations=['ReLU', 'ReLU'], learning_rate=0.01, alpha=0.002, 
                        nepochs=20, batch_size=100, loss='mean_squared_error', 
                        is_regression=True, noutputs=2, opt_config='SGD')
    
    assert mlp_tensorflow.noutputs == 2
    params = mlp_tensorflow.get_params()
    assert params['noutputs'] == 2
    
    mlp_tensorflow.fit(Xtr, ytr_multi)
    y_pred_tensorflow = mlp_tensorflow.predict(Xte)
    assert y_pred_tensorflow.shape == (yte_multi.shape[0], 2), f"Expected shape (50, 2), got {y_pred_tensorflow.shape}"


# def test_init_via_config(setup_teardown):
#     # make layer config file
#     layer_config = [('Linear', {
#         'units': 64,
#         'activation': 'relu'
#     }), ('Dropout', {
#         'p': 0.2
#     }), ('Linear', {
#         'units': 32,
#         'activation': 'relu'
#     }), ('Dropout', {
#         'p': 0.2
#     }), ('Linear', {
#         'units': 1,
#         'activation': 'linear'
#     })]
#     temp_path = setup_teardown
#     with open(os.path.join(temp_path, 'layers.config'), 'w') as f:
#         dump(layer_config, f)


#     opt = ['SGD', {'lr': 0.1, 'momentum': 0.9}]

#     mlp = MLP(
#         engine='pytorch',
#         nfeatures = 10,
#         regression=True,
#         nepochs=5,
#         batch_size=20,
#         loss='mse',
#         layer_config_file=os.path.join(temp_path, 'layers.config'),
#         opt_config=opt)
#     # Xtr, ytr, Xte, yte = data
#     # mlp.model.summary()
#     warnings.simplefilter("always")

