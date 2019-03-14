import pytest
import os
import warnings
import pkg_resources
import pandas as pd

from keras.layers import Input, Dense, Concatenate
from keras.models import Model
from keras.optimizers import Adam

from chemml.optimization import BEMCM
from chemml.datasets import load_organic_density


def model_creator_one_input(activation='relu', lr=0.001):
    # branch 1
    b1_in = Input(shape=(200,), name='inp1')
    b1_l1 = Dense(128, name='l1', activation=activation)(b1_in)
    b1_l2 = Dense(64, name='l2', activation=activation)(b1_l1)
    b1_l3 = Dense(32, name='l3', activation=activation)(b1_l2)
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
    b1_l1 = Dense(128, name='l1', activation=activation)(b1_in)
    b1_l2 = Dense(64, name='l2', activation=activation)(b1_l1)
    b1_l3 = Dense(32, name='l3', activation=activation)(b1_l2)
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
    al = BEMCM(
        model_creator=model_creator_one_input,
        U=features,
        target_layer='l3',
        train_size=50,
        test_size=50,
        batch_size=10)

def test_model_multiple_input():
    _, density, features = load_organic_density()
    al = BEMCM(
        model_creator = model_creator_two_inputs,
        U=features,
        target_layer = ['l3', 'b2_l1'],
        train_size=50,
        test_size=50,
        batch_size=10)

