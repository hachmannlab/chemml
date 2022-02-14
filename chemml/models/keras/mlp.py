import numpy as np
import pandas as pd  # required to load tensorflow properly

from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import SGD

from json import load
from importlib import import_module


class MLP(object):
    """
    Class associated with Multi-Layer Perceptron (Neural Network)

    Parameters
    ----------
    nhidden : int, optional, default: 1
        The number of hidden layers in the neural network (excluding input and output)

    nneurons: list, optional, default: [100] * nhidden
        The number of nodes in each hidden layer. Must be of same length as nhidden

    activations: list, optional, default: ['sigmoid'] * nhidden
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

    opt_config: list, optional, default: None
        optimizer configuration (for e.g., ["Adam",{"learning_rate":0.01}] or 
        ["SGD",{"lr":0.01, "momentum":0.9, "lr_decay":0.0, nesterov=False)
        Refer MLP test to see a sample file



    """

    def __init__(self,
                 nhidden=1,
                 nneurons=None,
                 activations=None,
                 learning_rate=0.01,
                 lr_decay=0.0,
                 nepochs=100,
                 batch_size=100,
                 loss='mean_squared_error',
                 regression=True,
                 nclasses=None,
                 layer_config_file=None,
                 opt_config=None):
        self.model = Sequential()
        if layer_config_file:
            self.layers = self.parse_layer_config(layer_config_file)
            self.nhidden = None
            self.nneurons = None
            self.activations = None
        else:
            self.layers = []
            self.nhidden = nhidden
            self.nneurons = nneurons if nneurons else [100] * nhidden
            self.activations = activations if activations else ['sigmoid'
                                                                ] * nhidden
        if opt_config:
            self.opt = self.parse_opt_config(opt_config)
        else:
            self.opt = SGD(
                lr=learning_rate, momentum=0.9, decay=lr_decay, nesterov=False)
        self.nepochs = nepochs
        self.batch_size = batch_size
        self.loss = loss
        self.is_regression = regression
        self.nclasses = nclasses
        
        
    def get_keras_model(self, include_output=True, n_layers=None):
        """
        Returns the entire Keras model or the model without the output layer in 
        its current state (fitted or compiled)
        
        Parameters
        __________
        include_output: bool
            if True it will return the entire model, if False it will return 
            the model without the output layer
            
        n_layers: int, optional (default=None)
            remove the last 'n' hidden layers from the model in addition to the output layer. 
            Note that this number should not include the output layer.
        
        Returns
        _______
        self.model: tensorflow.python.keras type object
            
        """
        if not include_output:
            head_model = Model(self.model.input, self.model.layers[-2].output)
            if n_layers is not None:
                if isinstance(n_layers, int):
                    head_model = Model(head_model.input, head_model.layers[-(1+n_layers)].output)
                else:
                    raise ValueError('n_layers should be an integer.')
            return head_model
        else:
            return self.model
   
    def save(self, path, filename):
        """
        Saves the chemml.models.MLP object along with the underlying 
        tensorflow.python.keras object
        
        Parameters
        __________
        path: str
            the path to the directory where the models should be saved
        filename: str
            the name of the model file without the file type
            
        """
        obj_dict = vars(self)
        self.model.save(path+'/'+filename+'.h5')
        # obj_dict['path_to_file'] = path +'/'+ filename+'.h5'
        obj_df = pd.DataFrame.from_dict(obj_dict,orient='index')
        obj_df.to_csv(path+'/'+filename+'_chemml_model.csv')
        print("File saved as "+path+"/"+filename+"_chemml_model.csv")
        
    def load(self, path_to_model):
        """
        Loads the chemml.models.MLP object along with the underlying 
        tensorflow.python.keras object
        
        Parameters
        __________
        path_to_model: str
            path to the chemml.models.MLP csv file
    
        """
        chemml_model = pd.read_csv(path_to_model,index_col=0)
        # self.model = load_model(chemml_model.loc['path_to_file'][0])
        self.model = load_model(path_to_model.split('_chemml_model.csv')[0] + '.h5')
        # optimizer config
        opt = self.model.optimizer.get_config()
        opt_list = [opt['name']]
        del opt['name']
        opt_list.append(opt)
        self.opt = self.parse_opt_config(opt_list)
        
        self.nepochs=int(chemml_model.loc['nepochs'][0])
        self.batch_size=int(chemml_model.loc['batch_size'][0])
        self.loss=chemml_model.loc['loss'][0]
        self.is_regression=eval(chemml_model.loc['is_regression'][0])
        self.nclasses=chemml_model.loc['nclasses'][0]
        
        if str(self.nclasses).lower() == 'nan':
            self.nclasses = None
        else:
            self.nclasses = int(self.nclasses)
            
        self.feature_size = int(chemml_model.loc['feature_size'][0])
        
        # layer config
        self.layers = [(n['class_name'],n['config']) for n in self.model.get_config()['layers']]
        self.nhidden = None
        self.nneurons = None
        self.activations = None
        
        return self

    def fit(self, X, y):
        """
        Train the MLP for training data X and targets y

        Parameters
        ----------
        X: array_like, shape=[n_samples, n_features]
            Training data

        y: array_like, shape=[n_samples,]
            Training targets

        """
        if len(self.layers) == 0:
            for i in range(self.nhidden):
                self.layers.append(('Dense', {
                    'units': self.nneurons[i],
                    'activation': self.activations[i]
                }))
            if self.is_regression:
                self.layers.append(('Dense', {
                    'units': 1,
                    'activation': 'linear'
                }))
            else:
                self.layers.append(('Dense', {
                    'units': self.nclasses,
                    'activation': 'softmax'
                }))
                
        self.feature_size = X.shape[-1]
        layer_name, layer_params = self.layers[0]
        layer_params['input_dim'] = X.shape[-1]
        keras_layer_module = import_module('tensorflow.keras.layers')
        for layer_name, layer_params in self.layers:
            layer = getattr(keras_layer_module, layer_name)
            self.model.add(layer(**layer_params))
        self.model.compile(loss=self.loss, optimizer=self.opt)
        self.batch_size = X.shape[
            0] if X.shape[0] < self.batch_size else self.batch_size
        self.model.fit(
            x=X, y=y, epochs=self.nepochs, batch_size=self.batch_size)

    def predict(self, X):
        """
        Return prediction for test data X

        Parameters
        ----------
        X: array_like, shape=[n_samples, n_features]
            Testing data

        Returns
        -------
        float
            Predicted value from model

        """
        return self.model.predict(
            X).squeeze() if self.is_regression else np.argmax(
                self.model.predict(X).squeeze())

    def score(self, X, y):
        """
        Predict results for test data X and compare with true targets y. Returns root mean square error if regression,
        accuracy if classification

        Parameters
        ----------
        X: array_like, shape=[n_samples, n_features]
            Test data

        y: array_like, shape=[n_samples,]
            True targets

        Returns
        -------
        float
            root mean square error if regression, accuracy if classification
        """
        prediction = self.model.predict(X).squeeze()
        if self.is_regression:

            return np.mean((prediction - y)**2)**0.5
        else:
            return np.sum(np.argmax(prediction, axis=1) == y) * 100. / len(y)

    def parse_layer_config(self, layer_config_file):
        """
        Internal method to parse a layer config file

        Parameters
        ----------
        layer_config_file: str
            Filepath that contains the layer configuration file - Refer MLP test to see a sample file
            Refer MLP test to see a sample file and https://keras.io/layers/about-keras-layers/
            for all possible types of layers and corresponding layer parameters

        Returns
        -------
        layers: list
            List of tuples containing layer type and dictionary of layer parameter arguments

        """
        with open(layer_config_file, 'r') as f:
            layers = load(f)
        return layers

    def parse_opt_config(self, opt_config):
        """
        Internal method to parse a optimizer config file

        Parameters
        ----------
        opt_config: list
            optimizer configuration (for e.g., ["Adam",{"learning_rate":0.01}] 
            or ["SGD",{"lr":0.01, "momentum":0.9, "lr_decay":0.0, nesterov=False)
            refer https://keras.io/optimizers/ for all possible types of 
            optimizers and corresponding optimizer parameters
    
        Returns
        -------
        opt: keras.optimizers
            keras optimizer created out of contents of optmizer configuration file

        """
        if isinstance(opt_config, list):
            opt_name, opt_params = opt_config[0], opt_config[1]
            
        keras_opt_module = import_module('tensorflow.keras.optimizers')
        opt = getattr(keras_opt_module, opt_name)(**opt_params)
        return opt
