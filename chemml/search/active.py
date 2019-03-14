"""
This module loads trained models to predict properties of organic molecules
"""

from __future__ import print_function
from builtins import range

import warnings
import types
import copy

import numpy as np
import pandas as pd
from keras import backend as K

from sklearn.model_selection import ShuffleSplit, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

class BEMCM(object):
    """
    The implementation of BEMCM for active learning of regression models.
    This algorithm assumes that you have a pool of unlabeled data points and a limited budget to label them.
    Thus we combine the efficiency of the machine learning models with our active learning approach to suggest
    optimal number of calculations by selected data points.

    The implementation of this algorithm follows an interactive approach.
    In other words, we often ask you to provide labels for the selected data points.

    Parameters
    ----------
    model_creator: FunctionType
        It's a function that returns the model. We call this function a couple of times during the search to build
        fresh models with random weights.
        Note that you should also compile your model inside the function. We don't provide options to compile the model.
        The compile (e.g., for Keras models) defines the loss function, the optimizer/learning rate, and the metrics.

    U: array-like
        The features/descriptors of unlabeled candidates that are available to be labeled.

    target_layer: str or list or FunctionType
        If str, it's the name of a layer of the Keras model that is linearly mapped to the outputs.
        If list, it's a list of str that each element corresponds to the name of layers.
        If a function, it should be able to receive a model that will be created using the 'model_creator' and the X inputs,
        and returns the outputs of the linear layer.

    train_size: float or int, optional (default = 0.1)
        It represents the absolute number of train samples that must be selected as the initial training set.
        The search will begin with this many training samples and labels are required immediately.
        Please choose a number based on:
            - your budget.
            - the minumum number that you think is enough to train your model.

    test_size: float or int, optional (default = 0.1)
        It represents the absolute number or proportion of test samples that will be held out for the evaluation of the model on
        all the stages of your active learning search.
        Note the test set will be acquired before search begins and won't be updated later during search.
        Please choose a number based on:
            - your budget.
            - the diversity of the pool of candidates (U).

    batch_size: int, optional (default = 10)
        The number of data points that this active learning algorithm selects after each training step.


    Attributes
    ----------

    train_indices: ndarray
        This is an array of all candidates' indices that are used as the training data.

    test_indices: ndarray
        This is an array of all candidates' indices that are used as the test data.

    seq_train_indices: list
        This is a list of arrays of training data indices in the same order that are queried.



    Notes
    -----
    - You won't be able to resume the search unless you deposit the labeled data if it's been requested.

    """

    def __init__(self, model_creator, U, target_layer, train_size=0.1, test_size=0.1, batch_size=10):
        self.model_creator = model_creator
        self.U = U
        self.target_layer = target_layer
        self.train_size = train_size
        self.test_size = test_size
        self.batch_size = batch_size
        self._fit()
        self._init_attributes()

    def _init_attributes(self):
        # available attributes
        self._queries = []
        self.train_indices = np.array([])
        self.seq_train_indices = []     # will become a list of lists (same order that are queried)
        self.test_indices = np.array([])
        self.U_indices = list(range(self.U.shape[0]))
        self.query_number = 0
        self._Y_train = None
        self._Y_test = None
        self._results = []
        self._random_results = []
        self.attributes = ['queries','train_indices','test_indices', 'seq_train_indices',
                           'query_number',
                           'X_train','X_test','Y_train','Y_test',
                           'results']

    def _X_train(self):
        return self.U[self.train_indices]

    def _X_test(self):
        return self.U[self.test_indices]

    @property
    def queries(self):
        return self._queries

    @property
    def X_train(self):
        return self._X_train()

    @property
    def X_test(self):
        return self._X_test()

    @property
    def Y_train(self):
        return self._Y_train

    @property
    def Y_test(self):
        return self._Y_test

    @property
    def results(self):
        results_header = ['num_query', 'num_training', 'num_test',
                               'mae', 'mae_std', 'rmse', 'rmse_std', 'r2', 'r2_std']
        df = pd.DataFrame(self._results, columns=results_header)
        return df

    @property
    def random_results(self):
        results_header = ['num_query', 'num_training', 'num_test',
                               'mae', 'mae_std', 'rmse', 'rmse_std', 'r2', 'r2_std']
        df = pd.DataFrame(self._random_results, columns=results_header)
        return df

    def _fit(self):
        # np array the input U
        self.U = np.array(self.U)
        self.U_size = self.U.shape[0]

        # Todo: support for sklearn linear models
        if not isinstance(self.model_creator, types.FunctionType):
            msg = "The parameter 'model' only accepts a keras model in the current version."
            raise TypeError(msg)

        # check int types
        if not isinstance(self.U_size, int) or not isinstance(self.train_size, int) or \
            not isinstance(self.test_size, int) or not isinstance(self.batch_size, int):
            msg = "The parameters 'U', 'train_size', 'test_size', and 'batch_size' must be int."
            raise TypeError(msg)

        # check the number of train and test sets
        if self.train_size >= self.U_size or self.test_size >= self.U_size or \
                (self.train_size+self.test_size) >= self.U_size:
            msg = "The train and test size and their sum must be less than the number of unlabeled data (U)"
            raise ValueError(msg)

    def get_target_layer(self, model, X):
        """
        The main function to get the latent features from the linear layer of the keras model.

        Returns
        -------
        ndarray
            The concatenated array of the specified hidden layers by parameter `target_layer`.
        """
        # inputs
        inp = model.input
        if isinstance(inp, list):
            if not isinstance(X, list):
                msg = "The input must be a list of arrays."
                raise ValueError(msg)
        else:
            if isinstance(X, list):
                msg = "Only one input array is required."
                raise ValueError(msg)
            else:
                # list of inp is required for K.function mapping
                inp = [inp]
                # if input is not a list should become a list
                X = [X]

        # outputs
        if isinstance(self.target_layer, str):
            out = [model.get_layer(self.target_layer).output]
        elif isinstance(self.target_layer, list):
            out = [model.get_layer(name).output for name in self.target_layer]
        else:
            msg = "The parameter 'linear_layer' must be str, list of str or a function."
            raise ValueError(msg)

        # define mapping function
        g = K.function(inp, out)

        # find and concatenate target layers
        target_layers = g(X)
        target_layers = np.concatenate(target_layers, axis=-1)

        return target_layers

    def initialize(self,random_state=90):
        """
        The function to initialize the training and test set for the search.
        You can run this function only once before starting the search.

        Parameters
        ----------
        random_state: int or RandomState, optional (default = 90)
            The random state will be directly passed to the sklearn.model_selection.ShuffleSplit
            extra info at: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html#sklearn-model-selection-shufflesplit

        Returns
        -------
        ndarray
            The training set indices (Python 0-index) from the pool of candidates (U).
            This is a 1D array.

        ndarray
            The test set indices (Python 0-index) from the pool of candidates (U).

        """
        if len(self._queries) != 0 or len(self.train_indices) != 0:
            train_indices = []
            test_indices = []
            for q_ind in self._queries:
                if q_ind[0] == 'initial training set':
                    train_indices = q_ind[1]
                elif q_ind[0] == 'test set':
                    test_indices = q_ind[1]
            if len(train_indices)>0 or len(test_indices)>0 :
                msg = "The class has been already initialized! You still have to deposit %i more training data and " \
                      "%i more test data. The same indices are returned respectively." % (len(train_indices), len(test_indices))
                warnings.warn(msg)
                return train_indices, test_indices
            else:
                msg = "The class has been already initialized and it can not be initialized again!"
                warnings.warn(msg)

        ss = ShuffleSplit(n_splits=1, test_size=self.test_size, train_size=self.train_size, random_state=random_state)
        for train_indices, test_indices in ss.split(range(len(self.U_indices))):
            self.U_indices = [i for i in self.U_indices if i not in train_indices and i not in test_indices]
            self._queries.append(['initial training set', train_indices])
            self.seq_train_indices.append(train_indices)
            self._queries.append(['test set', test_indices])
            return train_indices, test_indices

    def ignore(self, indices):
        """
        If you found out that the experimental setup or computational research on some of the candidates
        is not feasible, just pass a list of their indices here and we remove them from the list of queries.

        Parameters
        ----------
        indices: ndarray or list or tuple
            A 1D array of all the indices that should be removed from the list of `queries`.

        """
        indices = np.array(indices)
        # check the dimension of ind be one
        if indices.ndim != 1 :
            msg = "The dimension of 'indices' must be one, i.e., a list, tuple, or array of indices is requested."
            raise ValueError(msg)

        if len(self._queries) > 0:
            for query in self._queries:
                query[1] = np.array([i for i in query[1] if i not in indices])
        else:
            msg = "The list of queries is already empty."
            raise ValueError(msg)

    def deposit(self, indices, Y):
        """
        This function helps you to deposit the labeled data that was queried by initialize or search funstions.

        Parameters
        ----------
        indices: ndarray or list or tuple
            A 1D array of indices that was queried by initialize or search methods.
            You can deposit the data partially and it doesn't have to be the entire array that is queried.

        Y: array-like
            The 2-dimensional labels of the data points as it will be used for the training of the model.
            The first dimension of the array should be equal to the number of indices.
            Y must be at least 2 dimensional.

        Returns
        -------

        bool
            True, if deposited properly. False, otherwise.

        """
        indices = np.array(indices)
        Y = np.array(Y)
        # check the length of all is same
        if indices.shape[0] != Y.shape[0]:
            msg = "The first dimension of the input arrays should be equal to the number of indices."
            raise ValueError(msg)

        # check the length is not zero
        if indices.shape[0] == 0 :
            msg = "Received nothing to deposit!"
            print(msg)
            return False

        # check the dimension of ind be one
        if indices.ndim != 1 :
            msg = "The dimension of 'indices' must be one."
            raise ValueError(msg)

        # check if ind is a unique array of indices
        if len(set(indices)) != len(indices):
            msg = "The indices are not unique. This causes duplicate data and thus results in biased training."
            raise ValueError(msg)

        if len(self._queries) > 0:
            match_flag = False
            for query in self._queries:
                ind_in_q = np.where(np.in1d(indices, query[1], assume_unique=True))[0] # the position of matched numbers in the indices
                if len(ind_in_q)>0:
                    match_flag = True
                    if query[0] == 'test set':      # the only case that we query test data
                        if self._Y_test is None:
                            self._Y_test = Y[ind_in_q]
                        else:
                            self._Y_test = np.append(self._Y_test, Y[ind_in_q], axis=0)
                        # update test_indices
                        settled_inds = np.array(indices[ind_in_q])
                        self.test_indices = np.append(self.test_indices, settled_inds).astype(int)           # list of all indices
                    else:
                        if self._Y_train is None:
                            self._Y_train = Y[ind_in_q]
                        else:
                            self._Y_train = np.append(self._Y_train, Y[ind_in_q], axis=0)
                        # update test_indices
                        settled_inds = np.array(indices[ind_in_q])
                        self.train_indices = np.append(self.train_indices, settled_inds).astype(int)
                    # update q_ind and thus _queries
                    query[1] = np.array([i for i in query[1] if i not in settled_inds])
            if not match_flag:
                msg = "Can't match the indices with queries."
                print(msg)
                return False
            else:
                # update queries
                self._update_queries()
                return True
        else:
            msg = "The `queries` is empty. Can't deposit data if it's not been queried."
            print(msg)
            return False

    def _update_queries(self):
        """
        This function removes empty lists of queries.

        """
        self._queries = [q for q in self._queries if len(q[1])>0]

    def _scaler(self, scale):
        """
        The internal function to manage the X and Y scalers.

        Returns
        -------
        scaler
            The X scaler.

        scaler
            The Y scaler.

        """
        if isinstance(scale, list):
            if len(scale) == 2:
                Xscaler = scale[0]
                Yscaler = scale[1]
                return Xscaler, Yscaler
            else:
                msg = "The length of the parameter 'scale' must be two. The first scaler is for the X array and the second one for Y."
                raise ValueError(msg)
        else:
            if scale:
                return StandardScaler(), StandardScaler()
            else:
                return None, None

    def search(self, scale=True, n_evaluation=3, n_bootstrap=5, random_state=90, **kwargs):
        """
        The main function to start or continue an active learning search.
        The bootstrap approach is used to generate an ensemble of models that estimate the prediction
        distribution of the candidates' labels.

        Parameters
        ----------
        scale: bool or list, optional (default = True)
            if True, sklearn.preprocessing.StandardScaler will be used to scale X and Y before training.
            You can also pass a list of two scaler instances that perform sklearn-style fit_transform and transform methods
            for the X and Y, respectively.

        n_evaluation: int, optional (default = 3)
            number of times to repeat training of the model and evaluation on test set.

        n_bootstrap: int, optional (default = 5)
            The size of the ensemble based on bootstrapping approach.

        random_state: int or RandomState, optional (default = 90)
            The random state will be directly passed to the sklearn.model_selection.KFold
            extra info at: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html

        kwargs
            Any argument (except input data) that should be passed to the model's fit method.

        Returns
        -------
        ndarray
            The training set indices (Python 0-index) from the pool of candidates (U).
            This is a 1D array.

        """
        # check if queries are provided
        if len(self._queries) > 0 :
            msg = "The requested data must be provided first. Check the 'queries' attribute for the info regarding the indices of the queried candidates."
            print(msg)
            return False

        # get input data
        X_tr = self._X_train()
        X_te = self._X_test()
        Y_tr = copy.deepcopy(self._Y_train)
        Y_te = copy.deepcopy(self._Y_test)

        # scale
        X_scaler, Y_scaler = self._scaler(scale)
        if X_scaler is not None:
            # scale X arrays
            X_tr = X_scaler.fit_transform(X_tr)
            X_te = X_scaler.transform(X_te)
            # scale Y
            Y_tr = Y_scaler.fit_transform(Y_tr)

        # training and evaluation
        it_results = {'mae':[], 'rmse':[], 'r2':[]}
        Utr = X_scaler.transform(self.U[self.U_indices])    # unlabeled X values
        Y_U_pred_df = pd.DataFrame()  # empty dataframe to collect f(U) at each iteration
        learning_rate = []
        for it in range(n_evaluation):
            model = self.model_creator()
            model, Y_te_pred, mae, rmse, r2 = self._train_predict_evaluate(model, [X_tr,Y_tr,X_te],
                                                                           Y_scaler,
                                                                           Y_te,
                                                                           **kwargs)
            # Todo: how can we support multioutput?
            # predict Y of U, f(U)
            Y_U_pred_df[it] = Y_scaler.inverse_transform(model.predict(Utr)).reshape(-1,)

            # calculate the linear layer, phi(U)
            if it == 0:
                lin_layer = self.get_target_layer(model, Utr)
            else:
                temp = self.get_target_layer(model, Utr)
                lin_layer = lin_layer + temp

            # collect lr
            learning_rate.append(K.eval(model.optimizer.lr))

            # metrics
            it_results['mae'].append(mae)
            it_results['rmse'].append(rmse)
            it_results['r2'].append(r2)

        # store evaluation results
        results_temp = [self.query_number, len(self.train_indices), len(self.test_indices)]
        for metric in ['mae', 'rmse', 'r2']:
            results_temp.append(np.mean(np.array(it_results[metric]), axis=0))
            results_temp.append(np.std(np.array(it_results[metric]), axis = 0, ddof=0))
        self._results.append(results_temp)

        # find Y predictions of the remaining candidates U
        Y_U_pred = Y_U_pred_df.mean(axis=1).values.reshape(-1, 1)
        del Y_U_pred_df

        # find linear layer input
        lin_layer = lin_layer/float(n_evaluation)   # shape: (m,d)

        # avg of learning rates
        alpha = np.mean(learning_rate)
        print (alpha)

        # Bootstrap
        it = -1
        kf = KFold(n_splits=n_bootstrap, shuffle=True, random_state=random_state)
        for train_index, _ in kf.split(X_tr):
            Xtr = X_tr[train_index]
            Ytr = Y_tr[train_index]
            # model
            model = self.model_creator()
            model, Z_U_pred, _, _, _ = self._train_predict_evaluate(model,
                                                                    [Xtr,Ytr,Utr],
                                                                    Y_scaler,
                                                                    False,
                                                                    **kwargs)
            it += 1
            deviation = np.abs(Y_U_pred - Z_U_pred)     # shape: (m,1)
            # collect the bootstrap deviation from actual predictions
            if it==0:
                deviations = deviation
            else:
                deviations = np.append(deviations, deviation, axis=1)
            del Xtr, Z_U_pred
        del X_tr, Y_tr, Y_U_pred, Utr      # from now on we only need deviations and lin_layer

        # B_EMCM = EMCM - correlation_term
        # EMCM = mean(deviations * lin_layer
        # shapes: m = number of samples, d = length of latent features
        i_queries = []              # the indices of queries based on the length of Utr (not original U)
        correlation_term = {it: np.zeros(lin_layer.shape) for it in range(n_bootstrap)}   # shape of each: (m,d)
        while len(i_queries) < self.batch_size:
            norms = pd.DataFrame()  # store the norms of n_bootstrap models
            for it in range(n_bootstrap):
                # EMCM
                EMCM = deviations[:,it].reshape(-1,1) * lin_layer

                # correlation term acts after first selection
                if len(i_queries) > 0:
                    correlation_term[it] += self._correlation_term(i_queries[-1], deviations[:,it], lin_layer)

                # calculate batch EMCM
                B_EMCM = EMCM - alpha * correlation_term[it]    # shape: (m,d)

                # find norm and
                norm = np.linalg.norm(B_EMCM, axis=1)
                norms[it] = norm.reshape(-1,)

            # average and argmax of norms
            norms = norms.drop(i_queries, axis=0)   # remove previously selected points
            norms['mean'] = norms.mean(axis=1)
            norms['ind'] = norms.index
            norms.sort_values('mean', ascending=False, inplace=True)

            # memorize the initial ranking of the norms
            if len(i_queries) == 0:
                initial_ranking = list(norms.head(self.batch_size)['ind'])

            # select top candidate and update i_queries
            select = list(norms.head(1)['ind'])
            i_queries += select

        # increase global query number by one
        self.query_number += 1

        # make sure correlation term is making any difference than simple sorting of the initial norms
        if set(initial_ranking) <= set(i_queries):
            msg = "It seems that the correlation term is not effective. The initial ranking of the candidates are same as the final results. "
            warnings.warn(msg)

        # find original indices and update queries
        _queries = np.array([self.U_indices[i] for i in i_queries])
        self._queries.append(['batch #%i'%self.query_number, _queries])
        self.seq_train_indices.append(_queries)
        self.U_indices = [i for i in self.U_indices if i not in _queries]
        return _queries

    def _correlation_term(self, ind, deviation, lin_layer):
        """
        The internal function to find the correlation term
        """
        deviations_asterisk = deviation[ind] * lin_layer[ind]    # shape: (d,)
        deviations_asterisk_transpose = deviations_asterisk.reshape(-1,1).T  # shape (1,d)
        lin_layer_transpose = lin_layer.T       # shape: (d,m)
        dot = np.dot(deviations_asterisk_transpose, lin_layer_transpose)  # shape: (1,m)
        correlation_term_of_ind = dot.T * lin_layer       # shape (m,d) same shape as lin_layer
        return correlation_term_of_ind

    def _train_predict_evaluate(self, model=None, data_list=None, Y_scaler=None, metrics=None, **kwargs):
        """
        This internal function trains the model, predicts the test values, calculate metrics if requested.

        Returns
        -------
        class
            trained model

        ndarray
            predicted values

        float
            MAE

        float
            RMSE

        float
            R-squared
        """
        model.fit(data_list[0], data_list[1], **kwargs)
        preds = model.predict(data_list[2])
        preds = Y_scaler.inverse_transform(preds)

        mae=None; rmse=None; r2=None
        if isinstance(metrics, np.ndarray):
            mae = mean_absolute_error(metrics, preds)
            rmse = np.sqrt(mean_squared_error(metrics, preds))
            r2 = r2_score(metrics, preds)

        return model, preds, mae, rmse, r2

    def random_search(self, Y, scale=True, n_evaluation=10, random_state=90, **kwargs):
        """
        This function randomly select same number of data points as the active learning rounds and store the results.

        Parameters
        ----------
        Y: array-like
            The 2-dimensional label for all the candidates in the pool. Basically, you won't run this method unless you have the labels
            for all your samples. Otherwise, trust us and perform an active learning search.

        scale: bool or list, optional (default = True)
            if True, sklearn.preprocessing.StandardScaler will be used to scale X and Y before training.
            You can also pass a list of two scaler instances that perform sklearn-style fit_transform and transform methods
            for the X and Y, respectively.

        n_evaluation: int, optional (default = 3)
            number of times to repeat training of the model and evaluation on test set.

        random_state: int or RandomState, optional (default = 90)
            The random state will be directly passed to the sklearn.model_selection methods.
            Please find additional info at: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html

        kwargs
            Any argument (except input data) that should be passed to the model's fit method.

        Attributes
        ----------
        random_results: pandas dataframe
            The results from the random sampling to provide the baseline for your active learning search.
            You need to

        Notes
        -----
            - This method replicate the active learning training size with random sampling approach. Thus, you can run
            this function only if the results is not empty, i.e., you have run the active learning search at least once.


        """

        if self.query_number==0:
            msg = "You must run active learning search first."
            print(msg)
            return False
        # elif len(self._random_results)==0:
        #     self._random_results.append(self._results[0])

        Y = np.array(Y)
        if Y.shape[0] != self.U.shape[0]:
            msg = "The length of the Y array must be equal to the number of candidates in the U."
            print(msg)
            raise ValueError(msg)

        # find all indices except test indices
        except_test_inds = [i for i in range(len(self.U)) if i not in self.test_indices]

        # remaining training set size to run ML
        remaining_ind = [i for i in range(len(self._results)) if i not in range(len(self._random_results))]
        for ind in remaining_ind:
            # get training set size
            tr_size = self._results[ind][1] # the index 1 must be number of training data

            # training and evaluation
            it_results = {'mae': [], 'rmse': [], 'r2': []}

            # shuffle split, n_evaluation times
            ss = ShuffleSplit(n_splits=n_evaluation, test_size=None, train_size=tr_size,
                              random_state=random_state)
            for train_indices, _ in ss.split(except_test_inds):
                # training indices based on the original U
                actual_tr_inds = np.array(except_test_inds)[train_indices]
                X_tr = self.U[actual_tr_inds]
                Y_tr = Y[actual_tr_inds]

                # test set same as active learning
                X_te = self._X_test()
                Y_te = copy.deepcopy(self._Y_test)
                # scale
                X_scaler, Y_scaler = self._scaler(scale)
                if X_scaler is not None:
                    # scale X arrays
                    X_tr = X_scaler.fit_transform(X_tr)
                    X_te = X_scaler.transform(X_te)
                    # scale Y
                    Y_tr = Y_scaler.fit_transform(Y_tr)

                model = self.model_creator()
                model, Y_te_pred, mae, rmse, r2 = self._train_predict_evaluate(model, [X_tr, Y_tr, X_te],
                                                                               Y_scaler,
                                                                               Y_te,
                                                                               **kwargs)
                # metrics
                it_results['mae'].append(mae)
                it_results['rmse'].append(rmse)
                it_results['r2'].append(r2)

                # delete from memory
                del X_tr, X_te, model

            # store evaluation results
            results_temp = [self._results[ind][0], tr_size, len(self.test_indices)]
            for metric in ['mae', 'rmse', 'r2']:
                results_temp.append(np.mean(np.array(it_results[metric]), axis=0))
                results_temp.append(np.std(np.array(it_results[metric]), axis=0, ddof=0))
            self._random_results.append(results_temp)

        return True

