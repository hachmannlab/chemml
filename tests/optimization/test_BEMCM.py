import pytest
import os
import warnings
import pkg_resources
import pandas as pd
import numpy as np

from keras.layers import Input, Dense, Concatenate
from keras.models import Model
from keras.optimizers import Adam

from chemml.optimization import BEMCM
from chemml.datasets import load_organic_density


def model_creator_one_input(activation='relu', lr=0.001):
    # branch 1
    b1_in = Input(shape=(200,), name='inp1')
    b1_l1 = Dense(12, name='l1', activation=activation)(b1_in)
    b1_l2 = Dense(6, name='l2', activation=activation)(b1_l1)
    b1_l3 = Dense(3, name='l3', activation=activation)(b1_l2)
    # linear output
    out = Dense(1, name='outp', activation='linear')(b1_l3)
    ###
    model = Model(inputs=b1_in, outputs=out)
    adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0)
    model.compile(optimizer=adam,
                  loss='mean_squared_error',
                  metrics=['mean_absolute_error'])
    return model

def model_creator_two_inputs(activation='relu', lr = 0.001):
    # branch 1
    b1_in = Input(shape=(200, ), name='inp1')
    b1_l1 = Dense(12, name='l1', activation=activation)(b1_in)
    b1_l2 = Dense(6, name='l2', activation=activation)(b1_l1)
    b1_l3 = Dense(3, name='l3', activation=activation)(b1_l2)
    # branch 2
    b2_in = Input(shape=(200, ), name='inp2')
    b2_l1 = Dense(128, name='b2_l1', activation=activation)(b2_in)
    # merge branches
    merged = Concatenate(name='merged')([b1_l3, b2_l1])
    # linear output
    out = Dense(1, name='outp', activation='linear')(merged)
    ###
    model = Model(inputs = [b1_in, b2_in], outputs = out)
    adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0)
    model.compile(optimizer = adam,
                  loss = 'mean_squared_error',
                  metrics=['mean_absolute_error'])
    return model

def test_model_single_input():
    _, density, features = load_organic_density()
    # start with numpy array
    features = features.values
    density = density.values.reshape(-1,1)
    # instantiate the class
    al = BEMCM(
        model_creator=model_creator_one_input,
        U=features,
        target_layer='l3',
        train_size=50,
        test_size=40,
        batch_size=10)
    # initialize
    qtr, qte = al.initialize(random_state=7)
    assert len(qtr) == 50
    assert len(qte) == 40
    xtr = features[qtr]
    xte = features[qte]
    ytr = density[qtr]
    yte = density[qte]
    assert len(al.train_indices) == 0
    assert isinstance(al.U_indices, np.ndarray)
    assert len(al.U_indices) == features.shape[0]- 50 - 40
    # full deposit
    al.deposit(qtr, density[qtr])
    al.deposit(qte, density[qte])
    assert isinstance(al.train_indices, np.ndarray)
    assert len(al.train_indices) == 50
    assert isinstance(al.test_indices, np.ndarray)
    assert len(al.test_indices) == 40
    # check X and y
    assert xtr.shape == al.X_train.shape
    assert xtr[0][0] == al.X_train[0][0]
    assert xtr[-1][-1] == al.X_train[-1][-1]
    assert ytr[0][0] == al.Y_train[0][0]
    assert ytr[-1][0] == al.Y_train[-1][0]
    assert xte.shape == al.X_test.shape
    assert xte[0][0] == al.X_test[0][0]
    assert xte[-1][-1] == al.X_test[-1][-1]
    assert yte[0][0] == al.Y_test[0][0]
    assert yte[-1][-0] == al.Y_test[-1][0]
    # check orders
    inds = [i for i in range(density.shape[0]) if i not in al.train_indices and i not in al.test_indices]
    assert (density[inds].reshape(-1,) == density[al.U_indices].reshape(-1,)).all()
    assert density[al.U_indices].shape == density[inds].shape
    # search
    qtr = al.search(n_evaluation=1, n_bootstrap=1, verbose=0)
    assert len(qtr) == al.batch_size
    assert (al.queries[0][1]== qtr).all()
    assert al.U_indices.shape[0] == features.shape[0]- al.train_indices.shape[0] \
        - al.test_indices.shape[0] - al.batch_size
    # resutls
    assert isinstance(al.results, pd.DataFrame)
    # check U
    assert al.U.shape == features.shape
    assert (al.U == features).all()
    # random_search
    al.random_search(density, n_evaluation=1, verbose=0)
    assert al.results.shape == al.random_results.shape
    # deposit
    al.deposit(qtr, density[qtr])
    # check data once again
    assert al.X_train.shape == features[al.train_indices].shape
    assert al.X_train[0][0] == features[al.train_indices][0][0]
    # assert xtr[-1][-1] == al.X_train[-1][-1]

def test_model_multiple_input():
    _, density, features = load_organic_density()
    al = BEMCM(
        model_creator = model_creator_two_inputs,
        U=features,
        target_layer = ['l3', 'b2_l1'],
        train_size=50,
        test_size=50,
        batch_size=10)


    # initialize warning
    # warnings.simplefilter("always")
    # with warnings.catch_warnings(record=True) as w:
    #     al.initialize()
