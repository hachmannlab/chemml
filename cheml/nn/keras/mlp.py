from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

import numpy as np
from json import load
from importlib import import_module

class MLP(object):
    """
    Class associated with Multi-Layer Perceptron (Neural Network)    
    
    Parameters
    ----------
    nhidden : int, optional, default: 1
        The number of hidden layers in the neural network (excluding input and output)
        
    nneurons: list, optional, default: [100]
        The number of nodes in each hidden layer. Must be of same length as nhidden
        
    activations: list, optional, default: ['sigmoid']
        The activation type for each hidden layer. Must be of same length as nhidden.
        Refer https://keras.io/activations/ for list of valid activations
        
    nepochs: int, optional, default: 100
        Number of training epochs.
        
    batch_size: int, optional, default: 100
        Number of training samples in mini-batch
        
    loss: str, optional, default: 'mean_squared_error'
        Type of loss used to train the neural network.
        Refer https://keras.io/losses/ for list of valid losses
        
    regression: bool, optional, default: True
        Decides whether we are training for regression or classification task
        
    nclasses: int, optional, default: None
        Number of classes labels needs to be specified if regression is False
        
    layer_config_file: str, optional, default: None
        Path to the file that specifies layer configuration 
        Refer MLP test to see a sample file
        
    opt_config_file: str, optional, default: None
        Path to the file that specifies optimizer configuration
        Refer MLP test to see a sample file
    
    
    Attributes
    ----------
    
    Methods
    -------
    
    
    """
    def __init__(self, nhidden = 1, nneurons = [100], activations = ['sigmoid'],
                 learning_rate = 0.01, lr_decay = 0.0,
                 nepochs = 100, batch_size = 100, loss = 'mean_squared_error',
                 regression = True, nclasses = None,
                 layer_config_file = None, opt_config_file = None):
        self.model = Sequential()
        if layer_config_file:
            self.layers = self.parse_layer_config(layer_config_file)
        else:
            self.layers = []
            self.nhidden = nhidden
            self.nneurons = nneurons
            self.activations = activations
        if opt_config_file:
            self.opt = self.parse_opt_config(opt_config_file)
        else:
            self.opt = SGD(lr = learning_rate, momentum = 0.9, decay = lr_decay, nesterov = False)
        self.nepochs = nepochs
        self.batch_size = batch_size
        self.loss = loss
        self.is_regression = regression
        self.nclasses = nclasses

    def fit(self, X, y):
    """
    """    
        if len(self.layers) == 0:
            for i in xrange(self.nhidden):
                self.layers.append(('Dense',
                                    {'units': self.nneurons[i],
                                     'activation': self.activations[i]}))
            if self.is_regression:
                self.layers.append(('Dense',
                                    {'units': 1,
                                     'activation': 'linear'}))
            else:
                self.layers.append(('Dense',
                                    {'units': self.nclasses,
                                     'activation': 'softmax'}))
        layer_name, layer_params = self.layers[0]
        layer_params['input_dim'] = X.shape[-1]
        keras_layer_module = import_module('keras.layers')
        for layer_name, layer_params in self.layers:
            layer = getattr(keras_layer_module, layer_name)
            self.model.add(layer(**layer_params))
        self.model.compile(loss = self.loss, optimizer = self.opt)
        self.batch_size = X.shape[0] if X.shape[0] < self.batch_size else self.batch_size
        self.model.fit(x = X, y = y,  epochs = self.nepochs, batch_size = self.batch_size)

    def predict(self, X):
    """
    """
        return self.model.predict(X).squeeze() if self.regression else np.argmax(self.model.predict(X).squeeze())

    def score(self, X, y):
    """
    """
        prediction = self.model.predict(X).squeeze()
        if self.is_regression:
            return np.mean((prediction - y) ** 2) ** 0.5
        else:
            return np.sum(np.argmax(prediction, axis=1) == y) * 100. / len(y)

    def parse_layer_config(self, layer_config_file):
    """
    """
        with open(layer_config_file, 'r') as f:
            layers = load(f)
        return layers

    def parse_opt_config(self, opt_config_file):
    """
    """
        with open(opt_config_file, 'r') as f:
            opt_name, opt_params = load(f)
        keras_opt_module = import_module('keras.optimizers')
        opt = getattr(keras_opt_module, opt_name)(**opt_params)
        return opt


















