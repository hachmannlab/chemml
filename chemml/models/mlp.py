from multiprocessing.managers import ValueProxy
from multiprocessing.sharedctypes import Value
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F

import tensorflow as tf
tf.get_logger().setLevel(3) #to suppress warnings

from tensorflow import keras
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import SGD, Adam

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from collections import OrderedDict
from importlib import import_module

class pytorch_Net(nn.Module):
    '''Base class for custom pytorch DNN with forward function.

    Parameters
    ----------
    layers: list
        list of pytorch layers

    layer_config_file: list or None
        Layer config file to instantiate pytorch object with.
    
    '''
    def __init__(self, layers, layer_config_file):
        
        super(pytorch_Net,self).__init__()
        n = 0
        seq_l = []
        # print(layers)
        nn_module = import_module('torch.nn')
        if layer_config_file is None:
            for i in range(len(layers)-1): # 0 - 4 
                pt_layer = getattr(nn_module, layers[i][0])
                seq_l.append((str(n),pt_layer(layers[i][1]['units'],layers[i+1][1]['units'])))
                n = n+1
                try:
                    if layers[i][1]['activation'] != 'None' and layers[i+1][1]['units'] !=1:
                        # print(layers[i][1]['activation'],layers[i+1][1]['units'])
                        seq_l.append((str(n),getattr(nn_module, layers[i][1]['activation'])()))
                        n = n+1
                except:
                    raise ValueError('Incorrect Activation Format. Pytorch activation functions are case sensistive e.g., \
                                        ReLU not relu')
            # print(seq_l)
            self.base_model = nn.Sequential(OrderedDict(seq_l))
        else:
            self.base_model = nn.Sequential(*layers)

    def forward(self, X):
        """Forward propogation step.
        """
        return self.base_model(X)

class MLP(object):
    """
    Class associated with Multi-Layer Perceptron (Neural Network)

    Parameters
    ----------
    engine: str
        Determines the underlying ML library used to build the deep neural network
        can be either 'tensorflow' or 'pytorch'

    nfeatures: int
        number of input features

    nneurons: list, default = None
        The number of nodes in each hidden layer, required if layer_config_file is not provided

    activations: list, default = None
        The activation type for each hidden layer (len of activations should be the same as len of nneurons)
        required if layer_config_file is not provided
        Refer https://keras.io/activations/ for list of valid activations for tensorflow
        Refer https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity or 
        https://pytorch.org/docs/stable/nn.html#non-linear-activations-other for list of valid activations 
        for pytorch 
        e.g. ['ReLU','ReLU']

    nepochs: int, optional, default: 100
        Number of training epochs.

    batch_size: int, optional, default: 100
        Number of training samples in mini-batch
    
    alpha: float, default: 0.001 (defaults for pytorch: 0, keras: 0.01, sklearn: 0.0001)
        L2 regularization parameter. 
        If engine is pytorch, this will override the weight_decay parameter in the SGD optimizer
    
    loss: str, optional, default: 'mean_squared_error'
        Type of loss used to train the neural network.
        Refer https://keras.io/losses/ for list of valid losses for tensorflow
        Refer https://pytorch.org/docs/stable/nn.html#loss-functions for valid losses for pytorch

    regression: bool, optional, default: True
        Decides whether we are training for regression or classification task

    nclasses: int, optional, default: None
        Number of classes labels needs to be specified if regression is False

    layer_config_file: str, optional, default: None
        Path to the file that specifies layer configuration
        Refer MLP test to see a sample file
        Note: this variable SHOULD be consolidated with the layers variable to reduce redundancy

    opt_config: list or str, optional, default: 'sgd'
        optimizer configuration. 
        If str, should either be 'sgd' or 'adam'
        If list, should provide exact configurations and parameters corresponding to the respective engines, for e.g., 
        ["Adam",{"learning_rate":0.01}] or 
        ["SGD",{"lr":0.01, "momentum":0.9, "lr_decay":0.0, nesterov=False)]

    """
    def __init__(self, engine, nfeatures, nneurons=None, activations=None,
                learning_rate=0.01, nepochs=100, batch_size=100, alpha=0.001, loss='mean_squared_error', 
                is_regression=True, nclasses=None, layer_config_file=None, opt_config='sgd',random_seed=112, **params):

        if engine not in ['tensorflow','pytorch']:
            raise ValueError('engine has to be \'tensorflow\' or \'pytorch\'')

        self.engine = engine

        if layer_config_file == None:
            if nneurons == None or activations == None:
                raise TypeError('Either a layer_config_file or individual parameters for \
                                 nneurons, and activations are required.')
            elif len(nneurons) != len(activations):
                raise ValueError('No. of activations should be equal to the number of hidden layers \
                            (length of the nneurons list).')

        self.nfeatures = nfeatures
        self.nepochs = nepochs
        self.batch_size = batch_size
        self.alpha = alpha
        self.is_regression = is_regression
        self.nclasses = nclasses


        self.layer_config_file = layer_config_file
        self.opt_config = opt_config
        self.learning_rate = learning_rate
        self.nneurons = nneurons
        self.activations = activations
        self.loss = loss
        self.random_seed = random_seed

        torch.manual_seed(self.random_seed)
        tf.random.set_seed(self.random_seed)

        if self.is_regression:
            self.noutputs = 1
            self.output_activation = 'linear'
        else:
            self.noutputs = self.nclasses
            self.output_activation = 'softmax' 
               
        ############ TENSORFLOW ############
        if self.engine == 'tensorflow':
            ############ load_model ############
            if params:
                self.path_to_file = params['path_to_file']
                self.model = load_model(self.path_to_file)
                self.layers = self.model.layers
                self.opt = self.model.optimizer
            else:
                self._initialize_tensorflow()

        ############ PYTORCH ############
        elif self.engine == 'pytorch':
            if params: 
                self.path_to_file = params['path_to_file']
                self.layers = params['layers']
                self.losses = params['losses']
                self.model = pytorch_Net(self.layers,self.layer_config_file).base_model
                checkpoint = torch.load(self.path_to_file)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.opt = self._parse_opt_config(self.opt_config)
                self.opt.load_state_dict(checkpoint['optimizer_state_dict'])
            else:
                self._initialize_pytorch()

        


    def _initialize_tensorflow(self):
        self.model = Sequential()

        if self.layer_config_file:
            self.layers = self._parse_layer_config(self.layer_config_file)
            if isinstance(self.layers[0],tuple): 
                keras_layer_module = import_module('tensorflow.keras.layers')
                for i in range(len(self.layers)):
                    layer_name, layer_params = self.layers[i]
                    # if i == 0:
                    #     layer_params['input_shape'] = (self.nfeatures,)
                    
                    layer = getattr(keras_layer_module, layer_name)
                    self.model.add(layer(**layer_params))

                self.layers.insert(0,('Input',{'shape':(self.nfeatures,)}))
                # self.layer_config_file = self.layers 
            else:
                for i in self.layers:
                    self.model.add(i)
            
                self.nneurons = None
                self.activations = None
        else:
            self.layers = []
            # hidden layers
            self.model.add(keras.layers.Input(self.nfeatures))
            for i in range(len(self.nneurons)):
                self.layers.append(('Dense', {
                    'units': self.nneurons[i],
                    'activation': self.activations[i].lower(),
                    'kernel_initializer':'glorot_uniform',
                    'kernel_regularizer': keras.regularizers.l2(self.alpha),
                    }))

            # output layer
            self.layers.append(('Dense', {
                'units': self.noutputs,
                'activation': self.output_activation
            }))
            # Optionally, the first layer can receive an `input_shape`
            # model = tf.keras.Sequential()
            # model.add(tf.keras.layers.Dense(8, input_shape=(16,)))
            # Afterwards, we do automatic shape inference:
            # model.add(tf.keras.layers.Dense(4))

            # This is identical to the following:
            # model = tf.keras.Sequential()
            # model.add(tf.keras.Input(shape=(16,)))
            # model.add(tf.keras.layers.Dense(8))

            keras_layer_module = import_module('tensorflow.keras.layers')
            for i in range(len(self.layers)):
                layer_name, layer_params = self.layers[i]
                # if i == 0:
                #     layer_params['input_shape'] = (self.nfeatures,)
                
                layer = getattr(keras_layer_module, layer_name)
                self.model.add(layer(**layer_params))

            self.layers.insert(0,('Input',{'shape':(self.nfeatures,)}))

        if isinstance(self.opt_config, list) or isinstance(self.opt_config, tuple):
            self.opt = self._parse_opt_config(self.opt_config)
        elif self.opt_config.lower() == 'sgd':
            self.opt = SGD(learning_rate=self.learning_rate, momentum=0.9)
        elif self.opt_config.lower() == 'adam':
            self.opt = Adam(learning_rate=self.learning_rate)
        else:
            raise TypeError('opt_config should be a list/tuple or a str. If str, should either be "sgd" or "adam". If list, should provide exact configurations and parameters corresponding to the respective engines')
        
        self.model.compile(loss=self.loss, optimizer=self.opt)


    def _initialize_pytorch(self):
        
        if self.layer_config_file:
            self.layers = self._parse_layer_config(self.layer_config_file)
            self.nneurons = None
            self.activations = None
            # self.nneurons = [i[1]['units'] for i in self.layers if i[0].lower()!='dropout']
            # self.activations = [i[1]['activation'] for i in self.layers]
            
        else:
            self.layers = [('Linear',{'units':self.nfeatures, 'activation':self.activations[0]})]
            # self.nneurons = [self.nfeatures] + self.nneurons + [self.noutputs] # len = 5
            # self.nneu
            # self.activations = ['ReLU']+ self.activations + ['None']
            
            for i in range(len(self.nneurons)): #0,1,2
                self.layers.append(('Linear', {
                    'units': self.nneurons[i],
                    'activation': self.activations[i]
                }))
            # self.layer_config_file = self.layers
            self.layers.append(('Linear',{'units':1,'activation':None}))
        self.model = pytorch_Net(self.layers,self.layer_config_file).base_model

        if self.loss == 'mean_squared_error':
            self.loss = nn.MSELoss()

        # optimizer config
        if isinstance(self.opt_config, list) or isinstance(self.opt_config, tuple):
            self.opt = self._parse_opt_config(self.opt_config)
        elif self.opt_config.lower() == 'sgd':
            self.opt = torch.optim.SGD(self.model.parameters(),
                lr=self.learning_rate, momentum=0.9, weight_decay=self.alpha)
            self.opt_config = [self.opt_config,{'lr':self.learning_rate,'weight_decay':self.alpha,'momentum':0.9}]
        elif self.opt_config.lower() == 'adam':
            self.opt = torch.optim.Adam(self.model.parameters(),
                lr=self.learning_rate, weight_decay=self.alpha)
            self.opt_config = [self.opt_config,{'lr':self.learning_rate,'weight_decay':self.alpha}]
        else:
            raise TypeError('opt_config should be a list/tuple or a str. If str, should either be "sgd" or "adam". If list, should provide exact configurations and parameters corresponding to the respective engines')


    def _initialize_sklearn(self):
        pass


    def get_model(self, include_output=True, n_layers=None):
        """
        Returns the entire tensorflow or pytorch model in its current state (fitted or compiled)
        
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
        self.model: tensorflow.python.keras type object or torch.nn.Module type object 
        """

        ## need to test transfer learning
        if self.engine == 'pytorch':
            if n_layers is not None:
                if not isinstance(n_layers, int):
                    raise ValueError('n_layers should be an integer.')
                if n_layers == 0:
                    return self.model
                return self.model[:-(n_layers)]
            else:
                return self.model


    
        elif self.engine == 'tensorflow':
            if n_layers is None: 
                n_layers = 0
            if not isinstance(n_layers, int):
                raise ValueError('n_layers should be an integer.')
            if not include_output:
                for _ in range(n_layers + 1):
                    self.model.pop()

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
        required = ['engine','nfeatures','nepochs','batch_size','alpha',
                    'is_regression','nclasses','layer_config_file','opt_config',
                    'learning_rate','nneurons','activations','loss','random_seed']
        
        obj_dict = vars(self)
        chemml_options = {}
        for i in required:
            chemml_options[i] = obj_dict[i]
        
        if self.engine == 'tensorflow':
            self.model.save(path+'/'+filename+'.h5')
            chemml_options['path_to_file'] = path+'/'+filename+'.h5'
        elif self.engine == 'pytorch':
            checkpoint = {'model_state_dict':self.model.state_dict(),
                          'optimizer_state_dict':self.opt.state_dict()}
            torch.save(checkpoint, path+'/'+filename+'_checkpoint.pth')
            self.path_to_file = path+'/'+filename+'_checkpoint.pth'
            chemml_options['loss'] = str(self.loss)
            chemml_options['path_to_file'] = self.path_to_file
            # chemml_options['opt'] = self.opt
            chemml_options['losses'] = self.losses
            chemml_options['layers'] = self.layers
        import json
        with open(path+'/'+filename+'_chemml_model.json','w') as f:
            json.dump(chemml_options, f)
        print("File saved as "+path+"/"+filename+"_chemml_model.json")


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
        self.opt = self._parse_opt_config(opt_list)
        
        self.nepochs=int(chemml_model.loc['nepochs'][0])
        self.batch_size=int(chemml_model.loc['batch_size'][0])
        self.loss=chemml_model.loc['loss'][0]
        self.is_regression=eval(chemml_model.loc['is_regression'][0])
        self.nclasses=chemml_model.loc['nclasses'][0]
        
        if str(self.nclasses).lower() == 'nan':
            self.nclasses = None
        else:
            self.nclasses = int(self.nclasses)
            
        self.nfeatures = int(chemml_model.loc['nfeatures'][0])
        
        # layer config
        self.layers = [(n['class_name'],n['config']) for n in self.model.get_config()['layers']]
        self.nneurons = None
        self.activations = None
        
        return self


    def fit(self, X, y):
        if self.engine == 'pytorch':
            self._fit_pytorch(X, y)
        elif self.engine == 'tensorflow':
            self._fit_keras(X, y)


    def predict(self, X):
        if self.engine == 'pytorch':
            return self._predict_pytorch(X)
        elif self.engine == 'tensorflow':
            return self._predict_keras(X)


    def _fit_pytorch(self, X, y):

        if type(X) == torch.Tensor:
            pass
        elif type(X) == np.ndarray:
            X = torch.tensor((X),dtype=torch.float)
        else:
            raise TypeError('X has to be a numpy array or pytorch tensor')
        
        if type(y) == torch.Tensor:
            pass
        elif type(y) == np.ndarray:
            y = torch.tensor((y),dtype=torch.float)
        else:
            raise TypeError('y has to be a numpy array or pytorch tensor')
        
        self.losses = []

        #forward feed    
        for _ in range(self.nepochs):
            #shuffle rows
            permutation = torch.randperm(X.size()[0])
            # batch loss
            avg_loss=[]
            for i in range(0, X.size()[0], self.batch_size):
                #clear out the gradients from the last step loss.backward()
                self.opt.zero_grad()
                #creating indices for split
                indices = permutation[i:i+self.batch_size]
                #shuffle split done
                batch_x, batch_y = X[indices], y[indices]
                # y_pred = lin_model.forward(batch_x)
                y_pred = self.model.forward(batch_x)
                #calculate the loss
                loss = self.loss(y_pred, batch_y)
                #backward propagation: calculate gradients
                loss.backward()
                #update the weights
                self.opt.step()
                avg_loss.append(loss.item())

            # TODO: return self.losses to visualize learning process
            self.losses.append(np.mean(avg_loss))


    def _fit_keras(self, X, y):
        """
        Train the MLP for training data X and targets y

        Parameters
        ----------
        X: array_like, shape=[n_samples, n_features]
            Training data

        y: array_like, shape=[n_samples,]
            Training targets

        """ 
        self.batch_size = X.shape[0] if X.shape[0] < self.batch_size else self.batch_size
        self.model.fit(x=X, y=y, epochs=self.nepochs, batch_size=self.batch_size)


    def _predict_pytorch(self, X):
        """
        Return prediction for test data X

        Parameters
        ----------
        X: numpy.ndarray
            Testing data

        Returns
        -------
        y_hat: numpy.ndarray
            Model predictions for each data point in X

        """
        if type(X) == torch.Tensor:
            pass
        elif type(X) == np.ndarray:
            X = torch.tensor((X),dtype=torch.float)
        else:
            raise TypeError('X has to be a numpy array or pytorch tensor')

        y_hat = self.model(X)
        y_hat = y_hat.detach().numpy()
        return y_hat


    def _predict_keras(self, X):
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


    def _parse_layer_config(self, layer_config_file):
        """
        Internal method to parse a layer config file
        TODO: 1) layer config to json for display purposes 
        read text file - convert to json for all engines

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
        
            layers: list
            list of layer configurations
            pytorch:    nn.Linear(in_features=1, out_features=N_h, bias=True),

                        nn.Dropout(p=0.5), #50 % probability 
                        nn.ReLU(),
                        nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None),
                        nn.AvgPool2d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)
                        nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None) 

            keras: tf.keras.layers.Dropout(rate, noise_shape=None, seed=None, **kwargs)
                tf.keras.layers.Dense(units, activation=None,use_bias=True,kernel_initializer='glorot_uniform',
                                    bias_initializer='zeros',kernel_regularizer=regularizers.l2(alpha),bias_regularizer=None,
                                    activity_regularizer=None,kernel_constraint=None,bias_constraint=None)

                
            ## Sample Layer Config file

            # Pytorch

            # nn.Linear(in_features=1024, out_features=200, bias=True)
            # nn.Dropout(p=0.2)
            # nn.ReLU()
            # nn.Linear(in_features=200, out_features=100, bias=True)
            # nn.Dropout(p=0.2)
            # nn.ReLU()
            # nn.Linear(in_features=100, out_features=1, bias=True)

            self.layers, actual_layer_objects_list = self.parse_layer_config()

            for i in actual_layer_objects_list: model.append(i)
            model.compile()

            # Keras
            Dense(256, activation='relu', kernel_regularizer = regularizers.l2(alpha),
                            kernel_initializer=glorot_uniform(seed=1369), bias_initializer=glorot_uniform(seed=1369))
            Dropout(rate=0.2, noise_shape=None, seed=None, **kwargs)
            Dense(128, activation='relu', kernel_regularizer = regularizers.l2(alpha),
                            kernel_initializer=glorot_uniform(seed=1369), bias_initializer=glorot_uniform(seed=1369))
            Dropout(rate=0.2, noise_shape=None, seed=None, **kwargs)
            Dense(64, activation='relu', kernel_regularizer = regularizers.l2(alpha),
                            kernel_initializer=glorot_uniform(seed=1369), bias_initializer=glorot_uniform(seed=1369))
            Dense(32, activation='relu', kernel_regularizer = regularizers.l2(alpha),
                            kernel_initializer=glorot_uniform(seed=1369), bias_initializer=glorot_uniform(seed=1369))
            Dense(1, activation='linear', kernel_regularizer = regularizers.l2(alpha),
                            kernel_initializer=glorot_uniform(seed=1369), bias_initializer=glorot_uniform(seed=1369))
            
            l1_penalty = torch.nn.L1Loss(size_average=False)
            reg_loss = 0
            for param in model.parameters():
                reg_loss += l1_penalty(param)
                factor = const_val #lambda
                loss += factor * reg_loss
            
            for e.g.,layer_config = [('Linear', {'units': 64,'activation': 'relu'}), 
                                    ('Linear', {'units': 1,'activation': 'sigmoid'})]

        NO LAYER CONFIG FILE: neurons, activations, alpha, opt_config
    
    activations = nn.ModuleDict([
                ['lrelu', nn.LeakyReLU()],
                ['relu', nn.ReLU()]
    ])
    nn.Sequential(
        nn.Conv2d(in_f, out_f, *args, **kwargs),
        nn.BatchNorm2d(out_f),
        activations['relu']
    )
        WITH LAYER CONFIG: exact layer config file, opt_config
        """
        # ('Linear', {
        #             'units': self.nneurons[i],
        #             'activation': self.activations[i]
        #         })
        # if layers[i][1]['activation'].lower() == 'relu':
        #     # self.base_model.add_module(str(n),nn.ReLU())
        #     seq_l.append((str(n),nn.ReLU()))
        #     n = n+1
        # elif layers[i][1]['activation'].lower() == 'sigmoid':
        #     # self.base_model.add_module(str(n),nn.Sigmoid())
        #     seq_l.append((str(n),nn.Sigmoid()))
        #     n = n+1
        # from json import load as json_load
        # with open(layer_config_file, 'r') as f:
        #     layers = json_load(f) # json load function
        # return layers
        if isinstance(layer_config_file, list):
            return layer_config_file


    def _parse_opt_config(self, opt_config):
        """
        Internal method to parse a optimizer config file

        Parameters
        ----------
        opt_config: list
            optimizer configuration (for e.g., ["Adam",{"learning_rate":0.01}] 
            or ["SGD",{"lr":0.01, "momentum":0.9, "lr_decay":0.0, nesterov=False)]
            refer https://keras.io/optimizers/ for all possible types of 
            optimizers and corresponding optimizer parameters
    
        Returns
        -------
        opt: tensorflow.keras.optimizers or torch.optim
            optimizer created out of contents of optmizer configuration file

        """
        
        opt_name, opt_params = opt_config[0], opt_config[1]
        if self.engine == 'pytorch':
            opt_params['params'] = self.model.parameters()
            opt_params['weight_decay'] = self.alpha
            pytorch_opt_module = import_module('torch.optim')
            try:
                opt = getattr(pytorch_opt_module, opt_name.upper())(**opt_params)
                return opt
            except:
                raise ValueError('incorrect optimizer name or parameter for opt_config')
        elif self.engine == 'tensorflow':  
            keras_opt_module = import_module('tensorflow.keras.optimizers')
            try:
                opt = getattr(keras_opt_module, opt_name)(**opt_params)
                return opt
            except:
                raise ValueError('incorrect optimizer name or parameter for opt_config')


 

