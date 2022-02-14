import torch
from torch import nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 


class pytorch_Net(nn.Module):
    '''
    layers: list
        list of layer configurations 
        for e.g.,layer_config = [('Linear', {'units': 64,'activation': 'relu'}), 
                                 ('Linear', {'units': 1,'activation': 'sigmoid'})]
    '''
    def __init__(self,layers):
        super().__init__()
        self.model = nn.Sequential()
        n = 0
        #[input, hidden, hidden hidden, output]
        for i in range(len(layers)-1): # 0 - 4 
            if layers[i][0].lower() == 'linear':
                self.model.add_module(str(n),nn.Linear(layers[i][1]['units'],layers[i+1][1]['units']))
                n = n+1
            elif layers[i][0].lower() == 'dropout':
                self.model.add_module(str(n),nn.Dropout())
            # todo: multi-output
            if layers[i][1]['activation'].lower() == 'relu':
                self.model.add_module(str(n),nn.ReLU())
                n = n+1
            elif layers[i][1]['activation'].lower() == 'sigmoid':
                self.model.add_module(str(n),nn.Sigmoid())
                n = n+1
        
    
    # def forward(self, xdata):
    #     ydata = self.model(xdata)
    #     return ydata

class MLP_pytorch(object):
    """
    blabla
    """
    def __init__(self, nfeatures, nneurons=[100], activations=['sigmoid'],learning_rate=0.01, lr_decay=0.0, nepochs=100, batch_size=100, loss='mean_squared_error', regression=True, nclasses=None, layer_config_file=None, opt_config=None):
        
        self.nfeatures = nfeatures
        if len(nneurons) != len(activations):
            raise ValueError('No. of activations should be equal to the number of hidden layers (length of the nneurons list).')
        self.is_regression = regression
        self.nclasses = nclasses

        if self.is_regression:
            self.noutputs = 1
            output_activation = 'None'
        else:
            self.noutputs = self.nclasses
            output_activation = 'sigmoid'

        if layer_config_file:
            self.layers = self.parse_layer_config(layer_config_file)
            self.nneurons = [i[1]['units'] for i in self.layers if i[0].lower()!='dropout']
            self.activations = [i[1]['activation'] for i in self.layers]
        else:
            # self.layers = [('Linear',{'units':self.nfeatures,'activation':self.activations[0]})]
            self.layers = []
            self.nneurons = [self.nfeatures] + nneurons + [self.noutputs] # len = 5
            self.activations = ['None']+ activations + output_activation
            for i in range(len(self.nneurons)):
                self.layers.append(('Linear', {
                    'units': self.nneurons[i],
                    'activation': self.activations[i]
                }))

        #config file parser for pytorch config
        # if opt_config:
        #     self.opt = self.parse_opt_config(opt_config) 
        # else:
        #     self.opt = SGD(
        #         lr=learning_rate, momentum=0.9, decay=lr_decay, nesterov=False)

        self.nepochs = nepochs
        self.batch_size = batch_size
        self.loss = loss

        # model created
        self.model = pytorch_Net(self.layers) 

    def get_pytorch_model(self, include_output=True, n_layers=None):
        pass

    def save(self, path, filename):
        pass

    def load(self, path_to_model):
        pass

    def fit(self, X, y):
        losses = []
        y = torch.tensor((y),dtype=torch.float)
        X = torch.tensor((X),dtype=torch.float)

        #forward feed    
        for epochs in range(self.nepochs):
            
            #shuffle rows
            permutation = torch.randperm(X.size()[0])
            # batch loss
            avg_loss=[]
            for i in range(0,X.size()[0], self.batch_size):
                #clear out the gradients from the last step loss.backward()
                self.optimizer.zero_grad()
                #creating indices for split
                indices = permutation[i:i+self.batch_size]
                #shuffle split done
                batch_x, batch_y = X[indices], y[indices]
                # y_pred = lin_model.forward(batch_x)
                y_pred = self.model.model(batch_x)
                #calculate the loss
                loss = self.loss(y_pred, batch_y)
                #backward propagation: calculate gradients
                loss.backward()
                #update the weights
                self.optimizer.step()
                avg_loss.append(loss.item())
            losses.append(np.mean(avg_loss))

    def predict(self, X):
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
        X = torch.tensor((X),dtype=torch.float)
        y_hat = self.model.model(X)
        y_hat = y_hat.detach().numpy()
        return y_hat
    
    def parse_layer_config(self, layer_config_file):
        pass

    def parse_opt_config(self, opt_config):
        pass
