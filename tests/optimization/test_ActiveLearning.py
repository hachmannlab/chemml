import pytest
import os
import warnings
import pkg_resources
import pandas as pd
import numpy as np

from keras.layers import Input, Dense, Concatenate
from keras.models import Model
from keras.optimizers import Adam

from chemml.optimization import ActiveLearning
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

    # convert to numpy array
    features = features.values
    density = density.values.reshape(-1,1)

    # init exceptions
    with pytest.raises(TypeError):
        al = ActiveLearning(
            model_creator=2,
            U=features,
            target_layer='l3')
    with pytest.raises(TypeError):
        al = ActiveLearning(
            model_creator=model_creator_one_input,
            U=features,
            target_layer='l3',
            train_size=0.1)
    with pytest.raises(ValueError):
        al = ActiveLearning(
            model_creator=model_creator_one_input,
            U=features,
            target_layer='l3',
            train_size=1000)
    with pytest.raises(TypeError):
        al = ActiveLearning(
            model_creator=model_creator_one_input,
            U=features,
            target_layer='l3',
            train_size=1000,
            batch_size=[0,0,0])
    with pytest.raises(ValueError):
        al = ActiveLearning(
            model_creator=model_creator_one_input,
            U=features,
            target_layer='l3',
            train_size=1000,
            batch_size=[0,0,1])
    with pytest.raises(ValueError):
        al = ActiveLearning(
            model_creator=model_creator_one_input,
            U=features,
            target_layer='l3',
            train_size=1000,
            batch_size=[1,0,1],
            history=1)
    # test_type
    with pytest.raises(ValueError):
        al = ActiveLearning(
            model_creator=model_creator_one_input,
            U=features,
            target_layer='l3',
            train_size=50,
            test_type='fake')


    # instantiate the class
    al = ActiveLearning(
        model_creator=model_creator_one_input,
        U=features,
        target_layer='l3',
        train_size=50,
        test_size=40,
        batch_size=[2,1,2])

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
    assert len(al.U_indices) == features.shape[0]

    # reinitialize warnings
    warnings.simplefilter("always")
    with warnings.catch_warnings(record=True) as w:
        al.initialize()

    # deposit exception
    with pytest.raises(ValueError):
        al.deposit(qtr, density[qtr].reshape(-1,))
    with pytest.raises(ValueError):
        al.deposit(qtr[:-1], density[qtr])
    with pytest.raises(ValueError):
        al.deposit(np.array([[1]]), np.array([[1]]))
    with pytest.raises(ValueError):
        al.deposit(np.array([1,1]), np.array([[1],[2]]))
    with pytest.raises(ValueError):
        al.deposit(np.array([501,502]), np.array([[1],[2]]))

    # search w/ no deposit: exception
    with pytest.raises(ValueError):
        al.search()

    # full deposit
    assert al.deposit(qtr, density[qtr])
    assert al.deposit(qte, density[qte])
    assert len(al.queries) == 0

    # re deposit
    warnings.simplefilter("always")
    with warnings.catch_warnings(record=True) as w:
        al.deposit(qte, density[qte])

    assert isinstance(al.train_indices, np.ndarray)
    assert len(al.train_indices) == 50
    assert (al.train_indices == qtr).all()
    assert isinstance(al.test_indices, np.ndarray)
    assert len(al.test_indices) == 40

    # reinitialize exception
    with pytest.raises(ValueError):
        al.initialize()

    # check X and y
    assert (xtr == al.X_train).all()
    assert (ytr == al.Y_train).all()
    assert (xte== al.X_test).all()
    assert (yte== al.Y_test).all()

    # check orders
    inds = [i for i in range(density.shape[0]) if i not in al.train_indices and i not in al.test_indices]
    assert (density[inds].reshape(-1,) == density[al.U_indices].reshape(-1,)).all()
    assert density[al.U_indices].shape == density[inds].shape
    assert (al.Y_train == density[al.train_indices]).all()
    assert (al.Y_test == density[al.test_indices]).all()

    # search
    qtr = al.search(n_evaluation=1, ensemble='kfold', n_ensemble=1, verbose=0)

    assert (al.queries[0][1]== qtr).all()
    assert len(al.qbc_queries) == 1
    assert np.array([i not in al.train_indices for i in qtr]).all()
    assert np.array([i not in al.test_indices for i in qtr]).all()
    assert al.U_indices.shape[0] == features.shape[0]- al.train_indices.shape[0] \
        - al.test_indices.shape[0]

    # check X and y
    assert (xtr == al.X_train).all()
    assert (ytr == al.Y_train).all()
    assert (xte== al.X_test).all()
    assert (yte== al.Y_test).all()

    # check orders
    inds = [i for i in range(density.shape[0]) if i not in al.train_indices and i not in al.test_indices]
    assert (density[inds].reshape(-1,) == density[al.U_indices].reshape(-1,)).all()
    assert density[al.U_indices].shape == density[inds].shape
    assert (al.Y_train == density[al.train_indices]).all()
    assert (al.Y_test == density[al.test_indices]).all()

    # resutls
    assert isinstance(al.results, pd.DataFrame)
    assert al.results.shape[0] == 1

    # check U
    assert al.U.shape == features.shape
    assert (al.U == features).all()

    # random_search
    al.random_search(density, n_evaluation=1, verbose=0)
    assert al.results.shape == al.random_results.shape

    # check X and y
    assert (xtr == al.X_train).all()
    assert (ytr == al.Y_train).all()
    assert (xte== al.X_test).all()
    assert (yte== al.Y_test).all()

    # check orders
    inds = [i for i in range(density.shape[0]) if i not in al.train_indices and i not in al.test_indices]
    assert (density[inds].reshape(-1,) == density[al.U_indices].reshape(-1,)).all()
    assert density[al.U_indices].shape == density[inds].shape
    assert (al.Y_train == density[al.train_indices]).all()
    assert (al.Y_test == density[al.test_indices]).all()

    # deposit
    al.deposit(qtr, density[qtr])

    # check data once again
    assert (al.X_train == features[al.train_indices]).all()

    # search: ensemble exception
    with pytest.raises(ValueError):
        qtr = al.search(n_evaluation=1, ensemble='kfold', n_ensemble=0, normalize_internal=True, verbose=0)

    # search again
    qtr = al.search(n_evaluation=1, ensemble='kfold', n_ensemble=2, normalize_internal=True, verbose=0)
    assert (al.queries[0][1]== qtr).all()

    assert al.U_indices.shape[0] == features.shape[0]- al.train_indices.shape[0] \
        - al.test_indices.shape[0]

    # check data once again
    assert (al.X_train == features[al.train_indices]).all()
    assert (al.X_test == features[al.test_indices]).all()
    assert (al.Y_test == density[al.test_indices]).all()

    # check orders
    inds = [i for i in range(density.shape[0]) if i not in al.train_indices and i not in al.test_indices]
    assert (density[inds].reshape(-1,) == density[al.U_indices].reshape(-1,)).all()
    assert density[al.U_indices].shape == density[inds].shape
    assert (al.Y_train == density[al.train_indices]).all()
    assert (al.Y_test == density[al.test_indices]).all()

    # check unique
    assert len(np.unique(al.train_indices)) == len(al.train_indices)

    # visualize
    # plots = al.visualize(density)
    # assert len(plots) == 3
