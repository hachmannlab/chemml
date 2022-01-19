import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import Input, Dense, concatenate, multiply, Lambda
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras import regularizers, losses
from tensorflow.keras import backend as K

np.random.seed(69)

class LorentzLorenz():
    """
    A machine learning model for Lorentz-Lorenz (LL) estimates of refractive index.
    The model predicts refractive index, polarizability, and density of an organic molecule using its
    SMILES representation.

    The model is trained on 100K small organic molecules with their polarizabilities from DFT calculations, densities from
    molecular dynamics simulations, and refractive index by feeding calculated polarizabilities and densities into the
    LL model.
    """
    def __init__(self, n_features=4):
        
        self.n_features, self.n_targets = n_features, 3

        if self.n_features > 4 or self.n_features < 2:
            raise ValueError('Number of feature sets should be 2,3 or 4')

        self.seed = 69
        self.model_parameters = {
              'alpha':2.16e-4,
              'activation':'relu',
              'lr': 7.88e-6,
              'batch_size':128, 
              'epochs':10000,
              'verbose':0, 
              'validation_split':0.1
              }
        
        self.early_stopping = EarlyStopping(monitor='val_loss', min_delta=1e-8, patience=30, verbose=0, mode='auto')
        self.adam = Adam(lr=self.model_parameters['lr'], beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0)


        
    def preprocessing(self, features, targets, scaler_x=StandardScaler(), scaler_y=[StandardScaler(), StandardScaler(), StandardScaler()], test_ratio=0.1, random_state=69, shuffle=True, return_scaler=False):
        """
        This function removes constant columns from the features and scales them using the scalers provided by the user. It also scales the targets.
        
        Parameters
        ----------
        features: list of pandas DataFrames
            list of pandas DataFrames containing between 2 to 4 feature sets
        
        targets: list of pandas DataFrames
            list of pandas DataFrames for 3 individual target properties in the following order: 'refractive_index', 'polarizability', 'density'
        
        scaler_x: object
            scikit-learn scaler object (e.g., StandardScaler()) for features
        
        scaler_y: object
            scikit-learn scaler object (e.g., StandardScaler()) for targets
        
        test_ratio: float, default = 0.1
            float value between 0 and 1 (not inclusive) used for splitting data into training and test sets using scikit-learn train_test_split()
        
        random_state: int, default = 69
            random_state for scikit-learn train_test_split()
        
        shuffle: bool, default = True
            flag for shuffling the data before splitting it into training and test sets using scikit-learn train_test_split()
            
        return_scaler: bool
            if True a scaler_y will be returned for further use during model prediction
        
        Returns
        _______
        features_train: list
            list of numpy arrays containing scaled features for the training set
            
        features_test: list
            list of numpy arrays containing scaled features for the test set
            
        targets_train: list
            list of numpy arrays containing scaled targets for the training set
            
        targets_test: list
            list of numpy arrays containing scaled targets for the test set
            
        scaler_y: list of objects 
            if return_scaler is True, scaler_y is returned for inverse scaling of predicted targets
    
        """
        
        
        train_indices, test_indices = train_test_split(list(range(len(features[0]))), shuffle=shuffle, test_size=test_ratio,random_state=random_state)
        # test indices are not used during model creation/training 
        features_train, features_test, targets_train, targets_test = [], [], [], []     # targets.iloc[train_indices].values, targets.iloc[test_indices].values
        
        if len(features) != self.n_features:
            raise ValueError('length of the features list should be the same as n_features provided earlier')
            
        if len(targets) != self.n_targets:
            raise ValueError('length of the target list should be 3')
            
        else:
            # processing features
            for indx in range(len(features)): 
                #remove constant column
                features[indx] = features[indx].loc[:, (features[indx] != features[indx].iloc[0]).any()] # dropping constant columns
                features_train.append(scaler_x.fit_transform(features[indx].loc[train_indices].values))
                features_test.append(scaler_x.transform(features[indx].loc[test_indices].values))
                # todo try same scaler for all features
                
            # preprocessing targets
            for indx in range(len(targets)):
                targets_train.append(scaler_y[indx].fit_transform(targets[indx].loc[train_indices].values.reshape(-1,1))) 
                targets_test.append(targets[indx].loc[test_indices].values.reshape(-1,1))

            # targets_train = scaler_y.fit_transform(targets_train)
            
            if not return_scaler:
                return features_train, features_test, targets_train, targets_test
            else:
                return features_train, features_test, targets_train, targets_test, scaler_y

    def fit(self, X, Y):
        """
        Parameters
        __________
        X, Y: list of tensors
        
        Returns
        _______
        model: object
            fitted model object of type tensorflow.keras.models.Model
        """
        def create_model(X, activation, alpha):
            ##### First neural network #####
            nn1_in = Input(shape=(X[0].shape[1], ))
            nn1_l1 = Dense(256, activation=activation, kernel_regularizer = regularizers.l2(alpha),
                            kernel_initializer=glorot_uniform(seed=1369),
                            bias_initializer=glorot_uniform(seed=1369))(nn1_in)
            nn1_l2 = Dense(128, activation=activation, kernel_regularizer = regularizers.l2(alpha),
                            kernel_initializer=glorot_uniform(seed=1369), 
                            bias_initializer=glorot_uniform(seed=1369))(nn1_l1)
            nn1_l3 = Dense(64, activation=activation, kernel_regularizer = regularizers.l2(alpha),
                            kernel_initializer=glorot_uniform(seed=1369), 
                            bias_initializer=glorot_uniform(seed=1369))(nn1_l2)
            nn1_l4 = Dense(32, activation=activation, kernel_regularizer = regularizers.l2(alpha),
                            kernel_initializer=glorot_uniform(seed=1369), 
                            bias_initializer=glorot_uniform(seed=1369))(nn1_l3)
            # nn1_l4 = Dense(16, activation=activation, kernel_regularizer = regularizers.l2(alpha))(nn1_l3)
            nn1_l4_1 = Dense(2, activation='linear', kernel_regularizer = regularizers.l2(alpha),
                            kernel_initializer=glorot_uniform(seed=1369), 
                            bias_initializer=glorot_uniform(seed=1369))(nn1_l3)
            ###
            ##### Second neural network #####
            nn2_in = Input(shape=(X[1].shape[1], ))
            nn2_l1 = Dense(256, activation=activation, kernel_regularizer = regularizers.l2(alpha),
                            kernel_initializer=glorot_uniform(seed=1369), 
                            bias_initializer=glorot_uniform(seed=1369))(nn2_in)
            nn2_l2 = Dense(128, activation=activation, kernel_regularizer = regularizers.l2(alpha),
                            kernel_initializer=glorot_uniform(seed=1369), 
                            bias_initializer=glorot_uniform(seed=1369))(nn2_l1)
            nn2_l3 = Dense(64, activation=activation, kernel_regularizer = regularizers.l2(alpha),
                            kernel_initializer=glorot_uniform(seed=1369), 
                            bias_initializer=glorot_uniform(seed=1369))(nn2_l2)
            nn2_l4 = Dense(32, activation=activation, kernel_regularizer = regularizers.l2(alpha),
                            kernel_initializer=glorot_uniform(seed=1369), 
                            bias_initializer=glorot_uniform(seed=1369))(nn2_l3)
            # nn2_l4 = Dense(16, activation=activation, kernel_regularizer = regularizers.l2(alpha))(nn2_l3)
            nn2_l4_1 = Dense(2, activation='linear', kernel_regularizer = regularizers.l2(alpha),
                            kernel_initializer=glorot_uniform(seed=1369), 
                            bias_initializer=glorot_uniform(seed=1369))(nn2_l3)
            ###
            if self.n_features > 2:
                ##### Third neural network #####
                nn3_in = Input(shape=(X[2].shape[1], ))
                nn3_l1 = Dense(256, activation=activation, kernel_regularizer = regularizers.l2(alpha),
                                kernel_initializer=glorot_uniform(seed=1369), 
                                bias_initializer=glorot_uniform(seed=1369))(nn3_in)
                nn3_l2 = Dense(128, activation=activation, kernel_regularizer = regularizers.l2(alpha),
                                kernel_initializer=glorot_uniform(seed=1369), 
                                bias_initializer=glorot_uniform(seed=1369))(nn3_l1)
                nn3_l3 = Dense(64, activation=activation, kernel_regularizer = regularizers.l2(alpha),
                                kernel_initializer=glorot_uniform(seed=1369), 
                                bias_initializer=glorot_uniform(seed=1369))(nn3_l2)
                nn3_l4 = Dense(32, activation=activation, kernel_regularizer = regularizers.l2(alpha),
                                kernel_initializer=glorot_uniform(seed=1369), 
                                bias_initializer=glorot_uniform(seed=1369))(nn3_l3)
                # nn3_l4 = Dense(16, activation=activation, kernel_regularizer = regularizers.l2(alpha))(nn3_l3)
                nn3_l4_1 = Dense(2, activation='linear', kernel_regularizer = regularizers.l2(alpha),
                                kernel_initializer=glorot_uniform(seed=1369), 
                                bias_initializer=glorot_uniform(seed=1369))(nn3_l3)
                ###

            if self.n_features > 3: 
                ##### Fourth neural network #####
                nn4_in = Input(shape=(X[3].shape[1], ))
                nn4_l1 = Dense(256, activation=activation, kernel_regularizer = regularizers.l2(alpha),
                                kernel_initializer=glorot_uniform(seed=1369), 
                                bias_initializer=glorot_uniform(seed=1369))(nn4_in)
                nn4_l2 = Dense(128, activation=activation, kernel_regularizer = regularizers.l2(alpha),
                                kernel_initializer=glorot_uniform(seed=1369), 
                                bias_initializer=glorot_uniform(seed=1369))(nn4_l1)
                nn4_l3 = Dense(64, activation=activation, kernel_regularizer = regularizers.l2(alpha),
                                kernel_initializer=glorot_uniform(seed=1369), 
                                bias_initializer=glorot_uniform(seed=1369))(nn4_l2)
                nn4_l4 = Dense(32, activation=activation, kernel_regularizer = regularizers.l2(alpha),
                                kernel_initializer=glorot_uniform(seed=1369), 
                                bias_initializer=glorot_uniform(seed=1369))(nn4_l3)
                # nn4_l4 = Dense(16, activation=activation, kernel_regularizer = regularizers.l2(alpha))(nn4_l3)
                nn4_l4_1 = Dense(2, activation='linear', kernel_regularizer = regularizers.l2(alpha),
                                kernel_initializer=glorot_uniform(seed=1369), 
                                bias_initializer=glorot_uniform(seed=1369))(nn4_l3)
                                
            ###
            if self.n_features == 4:
                aNi = multiply([nn1_l4, nn2_l4, nn3_l4, nn4_l4]) ## second last layers are being multiplied (first layer of nn_RI)
                input_layers = [nn1_in, nn2_in, nn3_in, nn4_in]
                nn_pol_den_l1 = concatenate([nn1_l4_1, nn2_l4_1, nn3_l4_1, nn4_l4_1]) ## last layers are being concatenated (first layer of nn_pol_den)

            elif self.n_features == 3:
                aNi = multiply([nn1_l4, nn2_l4, nn3_l4]) ## second last layers are being multiplied (first layer of nn_RI)
                input_layers = [nn1_in, nn2_in, nn3_in]
                nn_pol_den_l1 = concatenate([nn1_l4_1, nn2_l4_1, nn3_l4_1]) ## last layers are being concatenated (first layer of nn_pol_den)

            elif self.n_features == 2:
                aNi = multiply([nn1_l4, nn2_l4]) ## second last layers are being multiplied (first layer of nn_RI)
                input_layers = [nn1_in, nn2_in]
                nn_pol_den_l1 = concatenate([nn1_l4_1, nn2_l4_1]) ## last layers are being concatenated (first layer of nn_pol_den)

            # nn_RI_l1 = Dense(1, activation='linear', kernel_regularizer = regularizers.l2(alpha))(aNi)
            # def riF(x):
            #     return K.sqrt((2 * x + 1) / (1 - x))
            # nn_RI_l2 = Lambda(riF,output_shape=(1,))(nn_RI_l1)
            ### 
            ##### fifth neural network (with second last layers from individual networks for RI) ######
            nn_RI_l1 = Dense(64, activation=activation, kernel_regularizer = regularizers.l2(alpha),
                            kernel_initializer=glorot_uniform(seed=1369), 
                            bias_initializer=glorot_uniform(seed=1369))(aNi)
            nn_RI_l2 = Dense(32, activation=activation, kernel_regularizer = regularizers.l2(alpha),
                            kernel_initializer=glorot_uniform(seed=1369), 
                            bias_initializer=glorot_uniform(seed=1369))(nn_RI_l1)
            RI_out1 = Dense(1, activation='linear', kernel_regularizer = regularizers.l2(alpha),
                            kernel_initializer=glorot_uniform(seed=1369), 
                            bias_initializer=glorot_uniform(seed=1369))(nn_RI_l2) ### RI 
            ###

            

            ##### output layers for polarizability and density ######
            pol_den_out2 = Dense(1, activation='linear', kernel_regularizer = regularizers.l2(alpha),
                            kernel_initializer=glorot_uniform(seed=1369), 
                            bias_initializer=glorot_uniform(seed=1369))(nn_pol_den_l1)
            pol_den_out3 = Dense(1, activation='linear', kernel_regularizer = regularizers.l2(alpha),
                            kernel_initializer=glorot_uniform(seed=1369), 
                            bias_initializer=glorot_uniform(seed=1369))(nn_pol_den_l1)
            ###

            ### model compilation
            model = Model(inputs=input_layers, outputs = [RI_out1, pol_den_out2, pol_den_out3])

            model.compile(optimizer = self.adam,
                        loss = 'mean_squared_error',
                        metrics=['mean_absolute_error'],
                        loss_weights = [1.,1.,1.]) ## weightage given to the loss calculated on each target property
            return model

        model = create_model(X, activation=self.model_parameters['activation'], 
                            alpha = self.model_parameters['alpha'])

        model.fit(x=X, y=Y, batch_size=self.model_parameters['batch_size'], epochs=self.model_parameters['epochs'],
                    verbose=self.model_parameters['verbose'],callbacks=[self.early_stopping],
                    validation_split=self.model_parameters['validation_split'])
                    
        return model
