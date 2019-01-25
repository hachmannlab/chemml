"""
This module loads trained models to predict properties of organic molecules
"""

from __future__ import print_function

import os
import pkg_resources
import numpy as np
import pandas as pd
from keras import backend as K
from keras.models import load_model
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect

from chemml.models.keras.trained.engine import check_array_input


class OrganicLorentzLorenz():
    """
    A machine learning model for Lorentz-Lorenz (LL) estimates of refractive index.
    The model predicts refractive index, polarizability, and density of an organic molecule using its
    SMILES representation.

    The model is trained on 100K small organic molecules with their polarizabilities from DFT calculations, densities from
    molecular dynamics simulations, and refractive index by feeding calculated polarizabilities and densities into the
    LL model.

    The model is a fully connected artificial neural network with 3 hidden layers. The number of neurons per layers from
    input layer to the output layer are as follow: 1024 --> 128 --> 64 --> 32 --> [1, 1, 1].
    """
    def __init__(self):
        self.path = pkg_resources.resource_filename('chemml', os.path.join('datasets', 'data', 'models',
                                                                           'keras', 'organic_lorentz_lorenz'))
        # load x and y scalers
        self.x_scaler = pd.read_csv(os.path.join(self.path, 'x_standard_scaler.csv'))
        self.y_scaler = pd.read_csv(os.path.join(self.path, 'y_standard_scaler.csv'))

    def load(self, summary=True):
        """
        This function loads the Keras model. The model consists of 3 hidden layers and more than 140K parameters.
        Parameters
        ----------
        summary: bool
            if True a summary of Keras model will be printed out.

        """
        self.model = load_model(os.path.join(self.path, 'Morgan_100k.h5'))
        if isinstance(summary, bool):
            if summary:
                self.model.summary()

    def __represent(self, smiles):
        # The descriptor must be a binary Morgan fingerprint with radius 2 and 1024 bits.

        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            msg = '%s is not a valid SMILES representation'%smiles
            raise ValueError(msg)
        else:
            return np.array(GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024))

    def predict(self, smiles, pprint=False):
        """
        After loading the model, this function predicts refractive index, polarizability, and density of the entery.

        Parameters
        ----------
        smiles: str
            The SMILES representaion of a molecule.

        pprint: bool
            If True, a short description of the predicted properties will be printed out.

        Returns
        -------
        tuple
            includes estimates of refractive index, polarizability, and density, respectively.

        """
        # Todo: smiles can be a list or file path?!
        # check smiles type
        if isinstance(smiles, str):
            # find descriptor
            self.descriptor = self.__represent(smiles)
        else:
            msg = "smiles must has `str` type."
            raise ValueError(msg)

        # preprocess fingerprint: keep all of them for this model
        xin = (self.descriptor - self.x_scaler['ss_mean'].values) / self.x_scaler['ss_scale'].values
        xin = xin.reshape(1, 1024)

        # y1: RI, y2: polarizability (Bohr^3), y3: density (Kg/m^3)
        y1, y2, y3 = self.model.predict(xin)
        ri = float(y1 * self.y_scaler['ss_scale'][0] + self.y_scaler['ss_mean'][0])
        pol = float(y2 * self.y_scaler['ss_scale'][1] + self.y_scaler['ss_mean'][1])
        den = float(y3 * self.y_scaler['ss_scale'][2] + self.y_scaler['ss_mean'][2])

        # print out predictions
        if pprint:
            print ('\ndata-driven model estimates:')
            print ('   LL refractive index:    ', '%.2f' % ri)
            print ('   polarizability (Bohr^3):', '%.2f' % pol)
            print ('   density (Kg/m^3):       ', '%.2f' % den)
        return (ri, pol, den)

    def train(self, X, Y, scale=True, kwargs_for_compile={}, kwargs_for_fit={}):
        """
        This function allows the user to retrain the model on a given data set for some further steps.
        Thus, all the parameters you are able to pass to a keras model's compile or fit methods can be passed to this
        function as well.

        Parameters
        ----------
        X: ndarray or dataframe
            If 2D array, must be with 1024 dimension and numerical type. It is recommended to be Morgan fingerprint representation of the molecules.
            If 1D array, must be an array of `str` type, each element represents a molecule in the SMILES format.
            If dataframe, it can be a 2D frame with one columnd of SMILES or 1024 columns of features.

        Y: list or dataframe
            a list of three numpy arrays for refractive index, polarizability, and density, respectively.
            The length of arrays must be same as the length of X.
            If dataframe, it must be a 2D frame with 3 columns, each for one of the properties.

        scale: bool, optional (default: True)
            If True the X and Y will be scaled in the same fashion as the original traning process (recommended).

        kwargs_for_compile: dict, optional (default: {})
            This dictionary could contain all the parameters that the compile method of keras models can receive.

        kwargs_for_fit: dict, optional (default: {})
            This dictionary could contain all the parameters that the fit method of keras models can receive.


        """
        # convert dataframe to ndarray
        if isinstance(X, pd.DataFrame):
            if X.ndim == 2 and X.shape[1] == 1:
                X = X.iloc[:,0].values
            elif X.ndim == 2 and X.shape[1] == 1024:
                X = X.values
            else:
                msg = "This function doesn't support the format of the input X."
                raise ValueError(msg)
        if isinstance(Y, pd.DataFrame):
            if Y.ndim == 2 and Y.shape[1] == 3:
                Y = [Y.iloc[:,0].values, Y.iloc[:,1].values, Y.iloc[:,2].values]
            else:
                msg = "This function doesn't support the format of the input Y."
                raise ValueError(msg)

        # check dimension of X
        itis, msg = check_array_input(X, 'X', 2, (None, 1024))
        if not itis:
            itis, msg = check_array_input(X, 'X', 1, (None,))
            if itis:
                X = np.array([self.__represent(i) for i in X])
            else:
                raise ValueError(msg)

        # check dimension of Y
        if isinstance(Y, list):
            if len(Y) == 3:
                if isinstance(Y[0], np.ndarray) and isinstance(Y[1], np.ndarray) and \
                        isinstance(Y[2], np.ndarray) and Y[0].ndim == Y[1].ndim == Y[2].ndim == 1:
                    if len(Y[0]) == len(Y[1]) == len(Y[2]) == len(X):
                        pass
                    else:
                        msg = "The length of all Y arrays and X must be same."
                        raise ValueError(msg)
                else:
                    msg = "All the Y arrays must be numpy 1D array of properties."
                    raise ValueError(msg)
            else:
                msg = "Y must contain 3 arrays."
                raise ValueError(msg)
        else:
            msg = "Y must be a list of arrays or a pandas dataframe."
            raise ValueError(msg)

        # scale
        if isinstance(scale, bool) and scale:
            # scale X
            xin = (X - self.x_scaler['ss_mean'].values) / self.x_scaler['ss_scale'].values
            xin = xin.reshape(X.shape[0], 1024)
            # scale Ys
            y1 = (Y[0] - self.y_scaler['ss_mean'][0]) / self.y_scaler['ss_scale'][0]
            y2 = (Y[1] - self.y_scaler['ss_mean'][1]) / self.y_scaler['ss_scale'][1]
            y3 = (Y[2] - self.y_scaler['ss_mean'][2]) / self.y_scaler['ss_scale'][2]
        else:
            msg = "The parameter scale must be boolean"
            raise ValueError(msg)

        # the actual compile and training
        from keras.optimizers import Adam
        adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0)
        default_kwargs = {'optimizer': adam,
                      'loss': 'mean_squared_error',
                      'metrics': ['mean_absolute_error']}
        default_kwargs.update(kwargs_for_compile)
        self.model.compile(**default_kwargs)
        self.model.fit(X, [y1, y2, y3], **kwargs_for_fit)

    def get_hidden_layer(self, X, id=1):
        """
        This functions return the first hidden layer of the model.

        Parameters
        ----------
        X: ndarray
            If 2D array, must be with 1024 dimension and numerical type. It is recommended to be Morgan fingerprint representation of the molecules.
            If 1D array, must be an array of `str` type, each element represents a molecule in the SMILES format.

        id: int
            This is the id of hidden layers. It can be any of 1, 2, or 3 for the first, second,
            or third hidden layer, respectively.

        Returns
        -------
        ndarray
            The array of shape (length_of_X, 128) as the outputs of the first hidden layer (id=1).
            The array of shape (length_of_X, 64) as the outputs of the first hidden layer (id=2).
            The array of shape (length_of_X, 32) as the outputs of the first hidden layer (id=3).
        """
        # check dimension of X
        itis, msg = check_array_input(X, 'X', 2, (None, 1024))
        if not itis:
            itis, msg = check_array_input(X, 'X', 1, (None,))
            if itis:
                X = np.array([self.__represent(i) for i in X])
            else:
                raise ValueError(msg)

        get_layer_output = K.function([self.model.layers[0].input],
                                          [self.model.layers[id].output])
        return get_layer_output([X])[0]

