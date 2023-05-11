import pytest
from chemml.datasets import load_organic_density
from chemml.models import MLP, TransferLearning
from chemml.utils import regression_metrics
import tensorflow as tf
import torch
from sklearn.preprocessing import StandardScaler
import numpy as np

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
    return Xtr, Xte, ytr, yte, scale_y



def test_tl_tensorflow(data):

    ################### CHILD MODEL ###################
    # initialize a ChemML MLP object
    tzvp_model = MLP(engine='tensorflow',nfeatures=200, nneurons=[32], activations=['ReLU'],
        learning_rate=0.01, alpha=0.001, nepochs=30, batch_size=1000, loss='mean_squared_error', 
        regression=True, nclasses=None, layer_config_file=None, opt_config='SGD')

    ################### PARENT MODEL ###################
    # initialize a ChemML MLP object
    mlp = MLP(engine='tensorflow',nfeatures=200, nneurons=[64,32,64], activations=['ReLU','ReLU','ReLU'],
        learning_rate=0.01, alpha=0.001, nepochs=20, batch_size=1000, loss='mean_squared_error', 
        regression=True, nclasses=None, layer_config_file=None, opt_config='SGD')
    
    Xtrain, Xtest, ytrain, ytest, scale_y = data

    # load pre-trained parent model
    mlp.fit(Xtrain, ytrain)
    y_mlp = scale_y.inverse_transform(mlp.predict(Xtest).reshape(-1,1))
    mae_small_model = regression_metrics(ytest,y_mlp)['MAE'].values[0]
    
    # initialize a TransferLearning object
    tl = TransferLearning(base_model=mlp)
    
    # transfer the hidden layers from parent model to child model and fit the model to the new data
    combined_model = tl.transfer(Xtrain, ytrain, tzvp_model)

    # predictions on test set
    y_pred = scale_y.inverse_transform(combined_model.predict(Xtest).reshape(-1,1))
    mae_large_model = regression_metrics(ytest,y_pred)['MAE'].values[0]
    
    combined_model.model.summary()
    #### TEST INCORRECT FEATURE SIZE ####
    # with pytest.raises(ValueError) as e:
    #     combined_model = tl.transfer(Wtrain, targets[:50], tzvp_model)
        
    # error_msg = e.value
    # assert 'No. of Features for new model should be the same as that of the base model' in str(error_msg)

    assert isinstance(combined_model, MLP)
    
    assert isinstance(y_pred, np.ndarray)
    assert mae_large_model < mae_small_model



def test_tl_pytorch(data):

    ################### CHILD MODEL ###################
    # initialize a ChemML MLP object
    tzvp_model = MLP(engine='pytorch',nfeatures=200, nneurons=[32], activations=['ReLU'],
        learning_rate=0.01, alpha=0.001, nepochs=30, batch_size=1000, loss='mean_squared_error', 
        regression=True, nclasses=None, layer_config_file=None, opt_config='SGD')

    ################### PARENT MODEL ###################
    # initialize a ChemML MLP object
    mlp = MLP(engine='pytorch',nfeatures=200, nneurons=[64,32,64], activations=['ReLU','ReLU','ReLU'],
        learning_rate=0.001, alpha=0.001, nepochs=30, batch_size=1000, loss='mean_squared_error', 
        regression=True, nclasses=None, layer_config_file=None, opt_config='SGD')
    
    Xtrain, Xtest, ytrain, ytest, scale_y = data

    # load pre-trained parent model
    mlp.fit(Xtrain, ytrain)
    y_mlp = scale_y.inverse_transform(mlp.predict(Xtest).reshape(-1,1))
    mae_small_model = regression_metrics(ytest,y_mlp)['MAE'].values[0]
    # initialize a TransferLearning object
    tl = TransferLearning(base_model=mlp, n_layers=0)

    # transfer the hidden layers from parent model to child model and fit the model to the new data
    combined_model = tl.transfer(Xtrain, ytrain, tzvp_model)

    # predictions on test set
    y_pred = scale_y.inverse_transform(combined_model.predict(Xtest).reshape(-1,1))
    mae_large_model = regression_metrics(ytest,y_pred)['MAE'].values[0]
    
    #### TEST INCORRECT FEATURE SIZE ####
    # with pytest.raises(ValueError) as e:
    #     combined_model = tl.transfer(Wtrain, targets[:50], tzvp_model)
        
    # error_msg = e.value
    # assert 'No. of Features for new model should be the same as that of the base model' in str(error_msg)

    assert isinstance(combined_model, MLP)
    
    assert isinstance(y_pred, np.ndarray)
    # assert mae_large_model < mae_small_model








