"""
This module provides interactive implementation of active learning algorithms to query optimal number of data points.
"""

from __future__ import print_function
from builtins import range

import warnings
import types
import copy

import numpy as np
import pandas as pd

from sklearn.model_selection import ShuffleSplit, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class ActiveLearning(object):
    """
    The implementation of active learning of regression models using BEMCM and QBC methods and approaches for distribution shift alleviations.
    This algorithm assumes that you have a pool of unlabeled data points and a limited budget to label them.
    Thus, we combine the efficiency of the machine learning models with our active learning approach to suggest
    optimal number of calculations to provide labeled data.

    The implementation of this algorithm follows an interactive approach.
    In other words, we often ask you to provide labels for the selected data points.

    Parameters
    ----------
    model_creator : FunctionType
        It's a function that returns the model. We call this function a couple of times during the search to build
        fresh models with random weights.
        Note that you should also compile your model inside the function. We don't provide options to compile the model.
        The compile (e.g., for Keras models) defines the loss function, the optimizer/learning rate, and the metrics.

    U : array-like
        The features/descriptors of unlabeled candidates that are available to be labeled.

    target_layer : str or list or FunctionType
        If str, it's the name of a layer of the Keras model that is linearly mapped to the outputs.
        If list, it's a list of str that each element corresponds to the name of layers.
        If a function, it should be able to receive a model that will be created using the 'model_creator' and the X inputs,
        and returns the outputs of the linear layer.

    train_size : int, optional (default = 100)
        It represents the absolute number of train samples that must be selected as the initial training set.
        The search will begin with this many training samples and labels are required immediately.
        Please choose a number based on:
            - your budget.
            - the minumum number that you think is enough to train your model.

    test_size : int, optional (default = 100)
        It represents the absolute number of test samples that will be held out for the evaluation of the model in all
        rounds of your active learning search.
        Note the test set will be acquired before search begins and won't be updated later during search.
        Please choose a number based on:
            - your budget.
            - the diversity of the pool of candidates (U).

    test_type : str, optional (default = 'passive')
        The value must be either 'passive' or 'active'.
        If passive, test set will be sampled randomly at the initialization.
        If active, test set will be sampled randomly at each round.

    batch_size : list, optional (default = [10])
        This is a list of maxumum three non-negative int values. Each value specifies the number of data points that
        our active learning approaches should query every round. The order of active learning approaches are as follows:
            - Batch Expected Model Change Maximization (BEMCM)
            - Query By Committee (QBC)
            - Distribution Shift Alleviation (DSA)

        Note that the last method (i.e., DSA) is a complement to the first two methods and can not be specified alone.

    history : int, optional (default = 2)
        This parameter must be an integer and greater than one. It specifies the number of previous active learning
        rounds to memorize for the distribution shift alleviation (DSA) approach.

    Attributes
    ----------
    queries : list
        This list provides information regarding the indices of queried candidates.
        for each element of the list:
            - The index-0 is a short description.
            - The index-1 is an array of indices.

    query_number : int
        The number of rounds you have run the active learning search method.

    U_indices : ndarray
        This is an array of the remaining unlabeled indices.

    train_indices : ndarray
        This is an array of all candidates' indices that are used as the training data.

    test_indices : ndarray
        This is an array of all candidates' indices that are used as the test data.

    Y_pred : ndarray
        The predicted Y values at the current stage. These values will be updated after each run of `search` method.

    results : pandas.DataFrame
        The final results of the active learning approach.

    random_results : pandas.DataFrame
        The final results of the random search.


    Methods
    -------
    initialize
    deposit
    search
    random_search
    visualize
    get_target_layer

    Notes
    -----
    - You won't be able to resume the search unless you deposit the requested labeled data.

    """

    def __init__(self, model_creator, U, target_layer, train_size=100, test_size=100,
                 test_type='passive', batch_size=[10], history=2):
        self.model_creator = model_creator
        self.U = U
        self.target_layer = target_layer
        self.train_size = train_size
        self.test_size = test_size
        self.test_type = test_type
        self.batch_size = batch_size
        self.history = history
        self._fit()

    def _X_train(self):
        """ We don't want to keep a potentially big matrix in the memory."""
        return self.U[self.train_indices]

    def _X_test(self):
        """ We don't want to keep a potentially big matrix in the memory."""
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
    def Y_pred(self):
        return self._Y_pred

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
            not isinstance(self.test_size, int):
            msg = "The parameters 'train_size' and 'test_size'must be int."
            raise TypeError(msg)

        # check batch size
        flag = False
        if isinstance(self.batch_size, list):
            if len(self.batch_size) <= 3:
                b = np.array([type(i)==int for i in self.batch_size])   # all int
                p = np.array([i>=0 for i in self.batch_size])   # all non-negative
                g = np.array([i>0 for i in self.batch_size])   # any greater than zero
                if b.all() and p.all() and g.any():
                    flag = True

        if not flag:
            msg = "The parameter'batch_size' must be a list of maximum 3 non-negative int values."
            raise TypeError(msg)
        elif len(self.batch_size)==3 and self.batch_size[2]>0 and sum(self.batch_size[:2])==0:
            msg = "The third value can not be the only positive value and it always accompany the first two methods."
            raise ValueError(msg)

        # check history
        flag = False
        if isinstance(self.history, int):
            if self.history > 1:
                flag = True
                self._history = np.zeros(self.U_size*self.history).reshape(self.U_size, self.history)

        if not flag:
            msg = "The parameter `history` must be a positive int and greater than 1."
            raise ValueError(msg)

        # check the number of train and test sets
        if self.train_size >= self.U_size or self.test_size >= self.U_size or \
                (self.train_size+self.test_size) >= self.U_size:
            msg = "The train and test size and their sum must be less than the number of unlabeled data (U)"
            raise ValueError(msg)

        # check the test_type value
        if self.test_type not in ('active', 'passive'):
            msg = "The parameter 'test_type' must be either 'active' or 'passive'."
            raise ValueError(msg)


        # other attributes
        self._queries = []
        self.qbc_queries = []
        self.dsa_queries = []
        self.bemcm_queries = []
        # all indices are numpy arrays
        self.train_indices = np.array([])
        self.test_indices = np.array([])
        self.initial_test_indices = np.array([])
        self.U_indices = np.array(range(self.U_size))
        self.query_number = 0
        self._Y_train = None
        self._Y_test = None
        self._Y_pred = None
        self._results = []
        self._random_results = []
        self.attributes = ['queries','train_indices','test_indices',
                           'query_number',
                           'X_train','X_test','Y_train','Y_test', 'Y_pred',
                           'results', 'random_results']

        self.lr = 0

    def get_target_layer(self, model, X):
        """
        The main function to get the latent features from the linear layer of the keras model.

        Returns
        -------
        target_layer : array-like
            The concatenated array of the specified hidden layers by parameter `target_layer`.
        """
        # import keras here
        from tensorflow.keras import backend as K

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
        random_state : int or RandomState, optional (default = 90)
            The random state will be directly passed to the sklearn.model_selection.ShuffleSplit
            extra info at: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html#sklearn-model-selection-shufflesplit

        Returns
        -------
        train_indices : array-like
            The training set indices (Python 0-index) from the pool of candidates (U).
            This is a 1D array.

        test_indices : array-like
            The test set indices (Python 0-index) from the pool of candidates (U).

        """
        if len(self._queries) != 0:
            train_indices = []
            test_indices = []
            for q_ind in self._queries:
                if q_ind[0] == 'test set':
                    test_indices = q_ind[1]
                else:
                    train_indices = q_ind[1]
            if len(train_indices)>0 or len(test_indices)>0 :
                msg = "The class has been already initialized! You still have to deposit %i more training data and " \
                      "%i more test data. The same indices are returned respectively." % (len(train_indices), len(test_indices))
                warnings.warn(msg)
                return train_indices, test_indices
        elif len(self.train_indices) != 0:
            msg = "The class has been already initialized and it can not be initialized again!"
            raise ValueError(msg)

        ss = ShuffleSplit(n_splits=1, test_size=self.test_size, train_size=self.train_size, random_state=random_state)
        for train_indices, test_indices in ss.split(range(len(self.U_indices))):
            self._queries.append(['initial training set', train_indices])
            self._queries.append(['test set', test_indices])
            return train_indices, test_indices

    def ignore(self, indices):
        """
        If you found out that the experimental setup or computational research on some of the candidates
        is not feasible, just pass a list of their indices here and we remove them from the list of queries.

        Parameters
        ----------
        indices : array-like
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
        This function helps you to deposit the data for candidates that were queried by initialize or search functions.

        Parameters
        ----------
        indices: array-like
            A 1-dimensional array of indices that was queried by initialize or search methods.
            You can deposit the data partially and it doesn't have to be the entire array that is queried.

        Y: array-like
            The 2-dimensional labels of the data points as it will be used for the training of the model.
            The first dimension of the array should be equal to the number of indices.
            Y must be at least 2 dimensional.

        Returns
        -------

        check : bool
            True, if deposited properly. False, otherwise.

        """
        indices = np.array(indices)
        Y = np.array(Y)
        # check if Y is 2 dimensional
        if Y.ndim != 2:
            msg = "The labels, 'Y', must be 2 dimensional."
            raise ValueError(msg)

        # check if the length of indices and labels are same
        if indices.shape[0] != Y.shape[0]:
            msg = "The first dimension of the input Y should be equal to the number of indices."
            raise ValueError(msg)

        # check the dimension of ind be one
        if indices.ndim != 1 :
            msg = "The dimension of 'indices' must be one: array, list, or tuple of indices would suffice."
            raise ValueError(msg)

        # check if indices are unique
        if len(set(indices)) != len(indices):
            msg = "The indices are not unique. This causes duplicate data and thus results in biased training."
            raise ValueError(msg)

        if len(self._queries) > 0:
            match_flag = False
            self.last_deposited_indices_ = np.array([])
            for query in self._queries:
                ind_in_q = np.where(np.in1d(indices, query[1], assume_unique=True))[0] # the position of matched numbers in the indices
                if len(ind_in_q)>0:
                    match_flag = True
                    if query[0] == 'test set':      # the only case that we query test data
                        if self._Y_test is None:
                            self._Y_test = Y[ind_in_q]
                        else:
                            self._Y_test = np.append(self._Y_test, Y[ind_in_q], axis=0)
                        # update test_indices and U_indices
                        settled_inds = np.array(indices[ind_in_q])
                        self.last_deposited_indices_ = np.append(self.last_deposited_indices_, settled_inds).astype(int)
                        self.test_indices = np.append(self.test_indices, settled_inds).astype(int)           # array of all indices
                        self.initial_test_indices = np.append(self.initial_test_indices, settled_inds).astype(int)
                        self.U_indices = np.array([i for i in self.U_indices if i not in settled_inds])
                    else:
                        if self._Y_train is None:
                            self._Y_train = Y[ind_in_q]
                        else:
                            self._Y_train = np.append(self._Y_train, Y[ind_in_q], axis=0)
                        # update train_indices and U_indices
                        settled_inds = np.array(indices[ind_in_q])
                        self.last_deposited_indices_ = np.append(self.last_deposited_indices_, settled_inds).astype(int)
                        self.train_indices = np.append(self.train_indices, settled_inds).astype(int)
                        self.U_indices = np.array([i for i in self.U_indices if i not in settled_inds])
                    # update q_ind and thus _queries
                    query[1] = np.array([i for i in query[1] if i not in settled_inds])
            if not match_flag:
                msg = "Can't match the indices with queries."
                raise ValueError(msg)
            else:
                msg = "we stored %i of passed indices. A list of them is in the 'last_deposited_indices_' attribute."%len(self.last_deposited_indices_)
                print(msg)
                # update queries
                self._update_queries()
                _ = self._update_train_test()
                return True
        else:
            msg = "The `queries` is empty. Can't deposit data if it's not been queried."
            warnings.warn(msg)
            return False

    def _update_queries(self):
        """
        This function removes empty lists of queries.

        """
        self._queries = [q for q in self._queries if len(q[1])>0]

    def _update_train_test(self):
        """
        This function take care of the test_type parameter.

        """
        if self.test_type == 'passive':
            return True
        if len(self._queries) > 0:
            return True
        else:
            # active test split
            all_indices = np.concatenate([self.train_indices, self.test_indices], axis=0)
            all_y = np.concatenate([self._Y_train, self._Y_test], axis=0)
            # select randomly
            ss = ShuffleSplit(n_splits=1, test_size=self.test_size, train_size=None, random_state=90)
            for train_indices, test_indices in ss.split(all_indices):
                # test
                self._Y_test = all_y[test_indices]
                self.test_indices = all_indices[test_indices]
                # train
                self._Y_train = all_y[train_indices]
                self.train_indices = all_indices[train_indices]

    def _scaler(self, scale):
        """
        The internal function to manage the X and Y scalers.

        Returns
        -------
        scaler_x : scaler
            The X scaler.

        scaler_y : scaler
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

    def search(self, n_evaluation=3, ensemble='bootstrap', n_ensemble=4, normalize_input=True, normalize_internal=False, random_state=90, **kwargs):
        """
        The main function to start or continue an active learning search.
        The bootstrap approach is used to generate an ensemble of models that estimate the prediction
        distribution of the candidates' labels.

        Parameters
        ----------
        n_evaluation : int, optional (default = 3)
            number of times to repeat training of the model and evaluation on test set.

        ensemble : str, optional (default = 'bootstrap')
            The sampling method to create n ensembles and estimate the predictive distributions.
                - 'bootstrap': standard bootstrap method (random choice with replacement)
                - 'shuffle' : sklearn.model_selection.ShuffleSplit
                - 'kfold' : sklearn.model_selection.KFold
            The 'shuffle' and 'kfold' methods draw samples that are smaller than training set.

        n_ensemble : int, optional (default = 5)
            The size of the ensemble based on bootstrapping approach.

        normalize_input : bool or list, optional (default = True)
            if True, sklearn.preprocessing.StandardScaler will be used to normalize X and Y before training.
            You can also pass a list of two scaler instances that perform sklearn-style fit_transform and transform methods
            for the X and Y, respectively.

        normalize_internal : bool, optional (default = False)
            if True, the internal variables for estimation of gradients will be normalized.

        random_state : int or RandomState, optional (default = 90)
            The random state will be directly passed to the sklearn.model_selection.KFold or ShuffleSplit
            Additional info at: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html

        kwargs
            Any argument (except input data) that should be passed to the model's fit method.

        Returns
        -------
        train_indices : array-like
            The training set indices (Python 0-index) from the pool of candidates (U).
            This is a 1D array.

        """
        # check if queries are provided
        if len(self._queries) > 0 :
            msg = "The requested data must be provided first. Check the 'queries' attribute for the info regarding the indices of the queried candidates."
            raise ValueError(msg)

        # assign batch size (length of batch size is not always 3, otherwise we could enumerate the list directly)
        bemcm = 0
        qbc = 0
        dsa = 0
        for i, s in enumerate(self.batch_size):
            if i==0:
                bemcm = s
            elif i==1:
                qbc = s
            elif i==2:
                dsa = s

        # get input data
        X_tr = self._X_train()
        X_te = self._X_test()
        Y_tr = copy.deepcopy(self._Y_train)
        Y_te = copy.deepcopy(self._Y_test)

        # scale
        X_scaler, Y_scaler = self._scaler(normalize_input)
        if X_scaler is not None:
            # scale X arrays
            Utr = X_scaler.fit_transform(self.U)
            X_tr = X_scaler.transform(X_tr)
            X_te = X_scaler.transform(X_te)
            # scale Y
            Y_tr = Y_scaler.fit_transform(Y_tr)
        else:
            Utr = self.U

        # assert not (Utr == self.U).all()  # run just for test

        # make sure the data is not overwritten
        # assert not (X_tr == self.U[self.train_indices]).all()   # run just for test
        assert not (Y_tr == self._Y_train).all()

        # training and evaluation
        it_results = {'mae':[], 'rmse':[], 'r2':[]}
        Y_U_pred_df = pd.DataFrame()  # empty dataframe to collect f(U) at each iteration
        learning_rate = []
        lin_layers = {}
        for it in range(n_evaluation):
            model = self.model_creator()
            model, _, mae, rmse, r2 = self._train_predict_evaluate(model,
                                                                           [X_tr, Y_tr, X_te],
                                                                           Y_scaler,
                                                                           Y_te,
                                                                           **kwargs)
            # Todo: how can we support multioutput?
            # predict Y of remaining U, f(Utr)
            if Y_scaler is not None:
                Y_U_pred_df[it] = Y_scaler.inverse_transform(model.predict(Utr)).reshape(-1,)

            # calculate the linear layer, phi(U), and collect lr for bemcm approach
            if bemcm:
                lin_layers[it] = self.get_target_layer(model, Utr[self.U_indices])
                # if it == 0:
                #     lin_layer = self.get_target_layer(model, Utr[self.U_indices])
                # else:
                #     temp = self.get_target_layer(model, Utr[self.U_indices])
                #     lin_layer = lin_layer + temp

                assert lin_layers[it].shape[0] == self.U_indices.shape[0]

                # collect lr
                from tensorflow.keras import backend as K
                learning_rate.append(K.eval(model.optimizer.lr))

            # metrics
            it_results['mae'].append(mae)
            it_results['rmse'].append(rmse)
            it_results['r2'].append(r2)
            del model

        # store evaluation results
        results_temp = [self.query_number, len(self.train_indices), len(self.test_indices)]
        for metric in ['mae', 'rmse', 'r2']:
            results_temp.append(np.mean(np.array(it_results[metric]), axis=0))
            results_temp.append(np.std(np.array(it_results[metric]), axis = 0, ddof=0))
        self._results.append(results_temp)

        # find Y predictions of all candidates U
        assert Y_U_pred_df.shape == (self.U_size, n_evaluation)
        self._Y_pred = Y_U_pred_df.mean(axis=1).values.reshape(-1, 1)
        # transform back to scaled
        if Y_scaler is not None:
            fU_preds_scaled = Y_scaler.transform(self._Y_pred)
        else:
            fU_preds_scaled = self._Y_pred
        assert self._Y_pred.shape == (self.U_size, 1)
        del Y_U_pred_df

        # find linear layer input and learning rate for bemcm approach
        if bemcm:
            best_ind = np.argmin(it_results['mae'])     # the order of lin_layer might be different from one model to another model
            lin_layer = lin_layers[best_ind]
            # lin_layer = lin_layer/float(n_evaluation)   # shape: (m,d)

            # scale linear layer
            if normalize_internal:
                scaler = StandardScaler()
                lin_layer = scaler.fit_transform(lin_layer)

            # avg of learning rates
            alpha = float(np.mean(learning_rate))
            self.lr = alpha

        # Ensemble
        if ensemble=='kfold' and n_ensemble>1:
            cv = KFold(n_splits=n_ensemble, shuffle=True, random_state=random_state)
            g = cv.split(X_tr)
        elif ensemble=='shuffle':
            cv = ShuffleSplit(n_splits=n_ensemble, train_size = X_tr.shape[0]-1, test_size= None, random_state=random_state)
            g = cv.split(X_tr)
        elif ensemble == 'bootstrap' or n_ensemble == 1:
            g = None
        else:
            msg = "You must select between 'bootstrap', 'kfold' or 'shuffle' sampling methods with the `n_ensemble` greater than zero."
            raise ValueError(msg)

        for it in range(n_ensemble):
            if g is None:
                train_index = np.random.choice(range(len(X_tr)), size=len(X_tr), replace=True)
            else:
                train_index, _ = next(g)
            Xtr = X_tr[train_index]
            Ytr = Y_tr[train_index]
            # model
            model = self.model_creator()
            model, Z_U_pred, _, _, _ = self._train_predict_evaluate(model,
                                                                    [Xtr,Ytr,Utr],
                                                                    None,   # don't inverse_transform preds
                                                                    False,
                                                                    **kwargs)
            deviation = fU_preds_scaled - Z_U_pred     # shape: (m,1)

            # collect the bootstrap deviation from actual predictions
            if it==0:
                deviations = deviation
            else:
                deviations = np.append(deviations, deviation, axis=1)

            del Xtr, Z_U_pred, model
        del X_tr, Y_tr, Utr      # from now on we only need deviations and lin_layer

        assert deviations.shape == (self.U_size, n_ensemble)

        # scale deviations
        if normalize_internal:
            scaler = StandardScaler()
            deviations = scaler.fit_transform(deviations)

        i_dsa_queries = []
        if dsa:
            i_dsa_queries = self._dsa_y_dist()
            # i_dsa_queries = self._dsa_test_y()
            # i_dsa_queries = self._dsa_unlabeled(deviations)

        i_qbc_queries = []
        if qbc and not bemcm:     # bemcm can cover for duplicates in all of the approaches
            i_qbc_queries = self._qbc(deviations[self.U_indices],i_dsa_queries)
        elif qbc:
            i_qbc_queries = self._qbc(deviations[self.U_indices],None)

        i_bemcm_queries = []
        if bemcm:
            i_bemcm_queries = self._bemcm(deviations[self.U_indices], lin_layer, alpha, normalize_internal,
                                              i_qbc_queries+i_dsa_queries)

        i_queries = np.unique(i_qbc_queries+i_dsa_queries+i_bemcm_queries)
        assert len(i_queries) == sum(self.batch_size)

        # increase global query number by one
        self.query_number += 1
        if qbc:
            self.qbc_queries += [self.U_indices[i] for i in i_qbc_queries]
        if dsa:
            self.dsa_queries += [self.U_indices[i] for i in i_dsa_queries if i not in i_qbc_queries]
        if bemcm:
            self.bemcm_queries += [self.U_indices[i] for i in i_bemcm_queries]

        # find original indices and update queries
        _queries = np.array([self.U_indices[i] for i in i_queries])

        self._queries.append(['batch #%i'%self.query_number, _queries])
        return _queries

    def _qbc(self, deviations, former_queries):
        """
        qbc approach
        """
        votes = pd.DataFrame(deviations)
        votes['sigma'] = deviations.std(axis=1)
        votes['ind'] = votes.index
        votes.sort_values('sigma', ascending=False, inplace=True)

        # check if qbc should compensate underselection by dsa approach
        if former_queries is None:
            qbc_size = self.batch_size[1]
            former_queries = []
        else:
            former_queries = np.unique(former_queries)
            qbc_size = sum(self.batch_size) - len(former_queries)

        # start query
        i_qbc_queries = []
        n = 1
        while len(i_qbc_queries) < qbc_size:
            select = list(votes.head(n)['ind'])
            while select[-1] in former_queries:
                n += 1
                select = list(votes.head(n)['ind'])
            i_qbc_queries += select[-1:]
            n += 1
        return i_qbc_queries

    def _dsa_y_dist(self):
        """
        dsa approach based on the distribution of the test data and predicted y values
        """
        # test distribuiton
        f = pd.DataFrame(self._Y_test, columns=['yt'])
        f['ind'] = f.index
        out, bins = pd.cut(f.yt, 100, retbins=True)
        groups = f.groupby(['ind', out])
        _ytest_dist = pd.DataFrame(groups.size().unstack().sum())
        _ytest_dist.sort_values(0, ascending=False, inplace=True)
        _ytest_dist['prob'] = _ytest_dist[0]/sum(_ytest_dist[0])

        # train distribution
        f = pd.DataFrame(self._Y_train, columns=['yt'])
        f['ind'] = f.index
        groups = f.groupby(['ind', pd.cut(f.yt, bins)])
        _ytrain_dist = pd.DataFrame(groups.size().unstack().sum())
        _ytrain_dist.sort_values(0, ascending=False, inplace=True)
        _ytrain_dist['prob'] = _ytrain_dist[0]/sum(_ytrain_dist[0])
        del f

        # difference in distribution
        self._dist_shift = pd.DataFrame(_ytest_dist['prob'] - _ytrain_dist['prob'])
        self._dist_shift.sort_values('prob', ascending=False, inplace=True)

        # find n selection per bin for underrepresented points in training data
        intervals = tuple(self._dist_shift[self._dist_shift['prob'] > 0].index)
        if len(intervals) == 0:
            return []   # if no distribution shift was diagnosed this method will be ineffective
        n_selection_p_bin = int(self.batch_size[2]/len(intervals))
        leftovers = self.batch_size[2] - len(intervals)*n_selection_p_bin

        # select based on unlabeled U and y preds
        unlabeled_y_preds = pd.DataFrame(self._Y_pred[self.U_indices], columns=['yp'])

        i_dsa_queries = []
        if n_selection_p_bin > 0:
            for i, bin in enumerate(intervals):
                indices = list(unlabeled_y_preds[(unlabeled_y_preds['yp'] > bin.left) & (unlabeled_y_preds['yp'] <= bin.right)].index)
                if len(indices) > n_selection_p_bin:
                    select = list(np.random.choice(indices, n_selection_p_bin, replace=False))
                else:
                    select = indices
                    leftovers += n_selection_p_bin - len(indices)
                i_dsa_queries += select

        # leftovers
        n = 0
        while len(i_dsa_queries) < self.batch_size[2]:
            bin = intervals[n]
            indices = list(unlabeled_y_preds[(unlabeled_y_preds['yp'] > bin.left) & (unlabeled_y_preds['yp'] <= bin.right)].index)
            if len(indices)> n_selection_p_bin+int(leftovers/len(intervals)) :
                select = np.random.choice(indices, 1)[0]
                while select in i_dsa_queries:
                    select = np.random.choice(indices, 1)[0]
            else:
                if n == len(intervals) - 1:
                    break
                else:
                    n+=1
                    continue
            i_dsa_queries.append(select)
            if n == len(intervals)-1:
                n=0
            else:
                n+=1

        assert len(i_dsa_queries) <= self.batch_size[2]

        return i_dsa_queries

    def _dsa_test_y(self):
        """
        dsa approach based on the test data and predicted y values
        """
        self._history_update(self._Y_pred.reshape(-1,))
        uncertainty_change = self._uncertainty_tracker(self._history[:,-1], self.history-1)      #shape: (m,)
        uncertainty_change_test = uncertainty_change[self.test_indices]
        votes = pd.DataFrame(uncertainty_change_test, columns=['uc'])
        votes['ind'] = votes.index
        votes.sort_values('uc', ascending=False, inplace=True)
        unlabeled_y_preds = self._Y_pred[self.U_indices]
        i_dsa_queries = []
        n = 1
        while len(i_dsa_queries) < self.batch_size[2]:
            select = list(votes.head(n)['ind'])
            idx = self.find_nearest(unlabeled_y_preds, self._Y_pred[self.test_indices[select[-1]]])
            n+=1
            if idx not in i_dsa_queries:
                i_dsa_queries.append(idx)
        return i_dsa_queries

    def find_nearest(self, array, value):
        array = array.reshape(-1,)
        idx = (np.abs(array - value)).argmin()
        return idx

    def _dsa_unlabeled(self,deviations):
        """
        dsa approach based on unlabeled data
        """
        dev = deviations.mean(axis=1)
        self._history_update(dev)
        uncertainty_change = self._uncertainty_tracker(self._history[:,-1], self.history-1)      #shape: (m,)
        uncertainty_change = uncertainty_change[self.U_indices]
        uncertainty_change = pd.DataFrame(uncertainty_change, columns=['uc'])
        uncertainty_change['ind'] = uncertainty_change.index
        uncertainty_change.sort_values('uc', ascending=False, inplace=True)
        i_dsa_queries = list(uncertainty_change.head(self.batch_size[2])['ind'])
        return i_dsa_queries

    def _uncertainty_tracker(self, dev, i):
        """
        recurcsive deviation of the history
        """
        i -= 1
        if i>= 0:
            dev = dev - self._history[:,i]
            dev = self._uncertainty_tracker(dev, i)
        return dev

    def _history_update(self, dev):
        """
        update last column of the history with dev and shift all the previous columns to the left.
        """
        for i in range(self.history-1):
            self._history[:,i] = self._history[:,i+1]
        self._history[:,-1] = dev

    def _bemcm(self, deviations, lin_layer, alpha, normalize_internal, former_queries):
        """
        bemcm approach
        B_EMCM = EMCM - correlation_term
        EMCM = mean(deviations * lin_layer)
        """
        former_queries = np.unique(former_queries)
        bemcm_size = sum(self.batch_size) - len(former_queries)
        # shapes: m = number of samples, d = length of latent features
        i_queries = []              # the indices of queries based on the length of U_indices (not original U)
        correlation_term = {it: np.zeros(lin_layer.shape) for it in range(deviations.shape[1])}   # shape of each: (m,d)
        while len(i_queries) < bemcm_size:
            norms = pd.DataFrame()  # store the norms of n_ensemble models
            for it in range(deviations.shape[1]):
                # EMCM
                EMCM = deviations[:,it].reshape(-1,1) * lin_layer

                # correlation term acts after first selection
                if len(i_queries) > 0:
                    correlation_term[it] += self._correlation_term(i_queries[-1], deviations[:,it], lin_layer, normalize_internal)

                # calculate batch EMCM
                B_EMCM = EMCM - alpha * correlation_term[it]    # shape: (m,d)

                # find norm and
                norm = np.linalg.norm(B_EMCM, axis=1)
                norms[it] = norm.reshape(-1,)

            # average and argmax of norms
            assert norms.shape == (self.U_indices.shape[0], deviations.shape[1])
            norms['mean'] = norms.mean(axis=1)
            norms['ind'] = norms.index
            norms = norms.drop(i_queries, axis=0)   # remove rows of previously selected points
            norms.sort_values('mean', ascending=False, inplace=True)

            # memorize the initial ranking of the norms
            if len(i_queries) == 0:
                initial_ranking = list(norms.head(bemcm_size)['ind'])

            # select top candidate and update i_queries
            n = 1
            select = list(norms.head(n)['ind'])
            while select[0] in former_queries:
                n += 1
                select = [list(norms.head(n)['ind'])[-1]]

            i_queries += select

        # make sure correlation term is making any difference than simple sorting of the initial norms
        if set(initial_ranking) <= set(i_queries):
            msg = "It seems that the correlation term is not effective. The initial ranking of the candidates are same as the final results."
            warnings.warn(msg)

        return i_queries

    def _correlation_term(self, ind, dev, lin_layer, normalize_internal):
        """
        The internal function to find the correlation term
        """
        deviations_asterisk = dev[ind] * lin_layer[ind]    # shape: (d,)
        deviations_asterisk_transpose = deviations_asterisk.reshape(-1,1).T  # shape (1,d)
        lin_layer_transpose = lin_layer.T       # shape: (d,m)
        dot = np.dot(deviations_asterisk_transpose, lin_layer_transpose)  # shape: (1,m)
        correlation_term_of_ind = dot.T * lin_layer       # shape (m,d) same shape as lin_layer
        if normalize_internal:
            scaler = StandardScaler()
            correlation_term_of_ind = scaler.fit_transform(correlation_term_of_ind)
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
        if Y_scaler is not None:
            preds = Y_scaler.inverse_transform(preds)

        mae=None; rmse=None; r2=None
        if isinstance(metrics, np.ndarray):
            mae = mean_absolute_error(metrics, preds)
            rmse = np.sqrt(mean_squared_error(metrics, preds))
            r2 = r2_score(metrics, preds)

        return model, preds, mae, rmse, r2

    def random_search(self, Y, test_type='passive', scale=True, n_evaluation=10, random_state=90, **kwargs):
        """
        This function randomly select same number of data points as the active learning rounds and store the results.

        Parameters
        ----------
        Y: array-like
            The 2-dimensional label for all the candidates in the pool. Basically, you won't run this method unless you have the labels
            for all your samples. Otherwise, trust us and perform an active learning search.

        test_type: str, optional (default = 'passive')
            The parameter value must be either 'passive' or 'active'.
            If passive, the initial randomly selected test set in the initialize method will be used for evaluation.
            If active, the current test set of active learning approach will be used for evaluation. Thus, if the test_type in active
            learning method is 'passive', you should run active and random search back to back and then deposit the data.
            This way you make sure both active and random search are tested on the same test sets.

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
        if Y.shape[0] != self.U_size:
            msg = "The length of the Y array must be equal to the number of candidates in the U."
            print(msg)
            raise ValueError(msg)

        # find all indices except test indices
        # we must consider the test type: active or passive
        if test_type == 'passive':
            test_indices = self.initial_test_indices
            except_test_inds = [i for i in range(self.U_size) if i not in self.initial_test_indices]
        elif test_type == 'active':
            test_indices = self.test_indices
            except_test_inds = [i for i in range(self.U_size) if i not in self.test_indices]
        else:
            msg = "The parameter 'test_type' must be either 'passive' or 'active'."
            raise ValueError(msg)

        # remaining training set size to run ML
        remaining_ind = [i for i in range(len(self._results)) if i not in range(len(self._random_results))]
        for ind in remaining_ind:
            # get training set size
            tr_size = self._results[ind][1] # the index 1 must be number of training data

            # training and evaluation
            it_results = {'mae': [], 'rmse': [], 'r2': []}

            # shuffle split, n_evaluation times
            ss = ShuffleSplit(n_splits=1, test_size=None, train_size=tr_size,
                              random_state=random_state)
            for train_indices, _ in ss.split(except_test_inds):
                # training indices based on the original U
                actual_tr_inds = np.array(except_test_inds)[train_indices]
                X_tr = self.U[actual_tr_inds]
                Y_tr = Y[actual_tr_inds]

                # test set based on the test type
                X_te = self.U[test_indices]
                Y_te = Y[test_indices]
                # scale
                X_scaler, Y_scaler = self._scaler(scale)
                if X_scaler is not None:
                    # scale X arrays
                    _ = X_scaler.fit_transform(self.U)
                    X_tr = X_scaler.transform(X_tr)
                    X_te = X_scaler.transform(X_te)
                    # scale Y
                    Y_tr = Y_scaler.fit_transform(Y_tr)

                for it in range(n_evaluation):
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
                    del model
                del X_tr, X_te, Y_tr
            # store evaluation results
            results_temp = [self._results[ind][0], tr_size, len(self.test_indices)]
            for metric in ['mae', 'rmse', 'r2']:
                results_temp.append(np.mean(np.array(it_results[metric]), axis=0))
                results_temp.append(np.std(np.array(it_results[metric]), axis=0, ddof=0))
            self._random_results.append(results_temp)

        return True

    def visualize(self, Y=None):
        """
        This function plot distribution of labels and principal components of the features for the last round of the
        active learning search.
        Note that this function uses the prediction values in the attribute `Y_pred`. This attribute will be updated after
        each round of search. Thus, we recommend you run `visualize` right after each call of search method to get a trajectory
        of the active learning process.

        Parameters
        ----------
        Y: array-like, optional (default = None)
            The 2-dimensional label for all the candidates in the pool (in case you have them!!!).
            If you have all the labels, we will be able to produce additional cool visualizations.


        Returns
        -------
        list
            A list of matplotlib.figure.Figure or tuples. This object contains information about the plot

        """
        if self.query_number == 0 and len(self._queries) > 0:
            msg = "Data is not available for visualization yet. You must deposit data first."
            raise ValueError(msg)

        collect_plots = {}

        # feature transformation
        pca = PCA(n_components=2)
        u = pca.fit_transform(self.U)   # use this (original) transformed feature space for all the X data
        u_rem = u[self.U_indices]
        # test is fixed
        xte = u[self.test_indices]
        # all trainings at the current state
        xtr = u[self.train_indices]
        if self.query_number == 0 or (self.query_number == 1 and len(self._queries) > 0):
            xtr_last_batch = u[self.train_indices][:self.train_size]
            ytr_last_batch = self._Y_train[:self.train_size]
        else:
            xtr_last_batch = u[self.train_indices][-sum(self.batch_size):]
            ytr_last_batch = self._Y_train[-sum(self.batch_size):]

        # plot1 : x/pc distribution
        collect_plots["dist_pc"] = self._visualize_dist_pc(u,u_rem, xte, xtr, xtr_last_batch)

        # plot2 : y distribution
        collect_plots["dist_y"] = self._visualize_dist_y(ytr_last_batch, Y)


        # plot3 : results' learning curves
        if len(self._results)>0:
            collect_plots["learning_curve"] = self._visualize_learning_curve()

        return collect_plots

    def _visualize_dist_pc(self, u, u_rem, xte, xtr, xtr_last_batch):
        """plot #1 : The distribution of PC1 values
        """

        import matplotlib
        matplotlib.use('Agg')
        import seaborn as sns
        import matplotlib.pyplot as plt

        # full
        fig1 = plt.figure()
        ax = sns.distplot(u[:, 0].reshape(-1, ), hist=False, color=sns.xkcd_rgb["denim blue"],
                          kde_kws={"shade": True, 'bw': 0.15}, label='Original U')  # label='Original U',
        sns.distplot(u_rem[:, 0].reshape(-1, ), hist=False, color=sns.xkcd_rgb["tangerine"],
                     kde_kws={"shade": True, 'bw': 0.15}, ax=ax, label='Current U')  # label='Current U',
        sns.distplot(xtr[:, 0].reshape(-1, ), hist=False, color=sns.xkcd_rgb["medium green"],
                     kde_kws={"shade": True, 'bw': 0.15}, ax=ax, label='Entire Train')  # label='Entire Train',
        sns.distplot(xte[:, 0].reshape(-1, ), hist=False, color=sns.xkcd_rgb["pale red"],
                     kde_kws={"shade": True, 'bw': 0.15}, ax=ax, label='Test')  # label='Entire Test',
        if sum(self.batch_size) != 1:
            sns.distplot(xtr_last_batch[:, 0].reshape(-1, ), hist=False, color=sns.xkcd_rgb["light purple"],
                     kde_kws={"shade": True, 'bw': 0.15}, ax=ax, label='Last Batch')  # label='Last Batch',
        # labels
        ax.set_ylabel(r'$\pi(PC1)$')
        ax.set_xlabel(r'$Features-PC1$')

        # font size
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(14)
        
        
        # all except last batch
        fig2 = plt.figure()
        ax2 = sns.distplot(u[:, 0].reshape(-1, ), hist=False, color=sns.xkcd_rgb["denim blue"],
                          kde_kws={"shade": True, 'bw': 0.15}, label='Original U')  # label='Original U',
        sns.distplot(u_rem[:, 0].reshape(-1, ), hist=False, color=sns.xkcd_rgb["tangerine"],
                     kde_kws={"shade": True, 'bw': 0.15}, ax=ax2, label='Current U')  # label='Current U',
        sns.distplot(xtr[:, 0].reshape(-1, ), hist=False, color=sns.xkcd_rgb["medium green"],
                     kde_kws={"shade": True, 'bw': 0.15}, ax=ax2, label='Entire Train')  # label='Entire Train',
        sns.distplot(xte[:, 0].reshape(-1, ), hist=False, color=sns.xkcd_rgb["pale red"],
                     kde_kws={"shade": True, 'bw': 0.15}, ax=ax2, label='Test')  # label='Entire Test',

        # labels
        ax2.set_ylabel(r'$\pi(PC1)$')
        ax2.set_xlabel(r'$Features-PC1$')

        # font size
        for item in ([ax2.title, ax2.xaxis.label, ax2.yaxis.label] +
                     ax2.get_xticklabels() + ax2.get_yticklabels()):
            item.set_fontsize(14)


        # all except Us
        fig3 = plt.figure()
        ax3 = sns.distplot(xtr[:, 0].reshape(-1, ), hist=False, color=sns.xkcd_rgb["medium green"],
                     kde_kws={"shade": True, 'bw': 0.15}, label='Entire Train')  # label='Entire Train',
        sns.distplot(xte[:, 0].reshape(-1, ), hist=False, color=sns.xkcd_rgb["pale red"],
                     kde_kws={"shade": True, 'bw': 0.15}, ax=ax3, label='Test')  # label='Entire Test',
        if sum(self.batch_size) != 1:
            sns.distplot(xtr_last_batch[:, 0].reshape(-1, ), hist=False, color=sns.xkcd_rgb["light purple"],
                     kde_kws={"shade": True, 'bw': 0.15}, ax=ax3, label='Last Batch')  # label='Last Batch',

        # labels
        ax3.set_ylabel(r'$\pi(PC1)$')
        ax3.set_xlabel(r'$Features-PC1$')

        # font size
        for item in ([ax3.title, ax3.xaxis.label, ax3.yaxis.label] +
                     ax3.get_xticklabels() + ax3.get_yticklabels()):
            item.set_fontsize(14)

        return (fig1, fig2, fig3)

    def _visualize_dist_y(self, ytr_last_batch, Y=None):
        """plot #2 : The distribution of y values
        """

        import matplotlib
        matplotlib.use('Agg')
        import seaborn as sns
        import matplotlib.pyplot as plt

        # full
        fig1 = plt.figure()
        ax = sns.distplot(self._Y_pred.reshape(-1, ), hist=False, color=sns.xkcd_rgb["tangerine"],
                           kde_kws={"shade": True, 'bw': 0.15}, label='Predicted U')  # label='Predicted U',
        if Y is not None:
            sns.distplot(Y.reshape(-1, ), hist=False, color=sns.xkcd_rgb["denim blue"],
                         kde_kws={"shade": True, 'bw': 0.15}, ax=ax, label='Labeled U')  # label='Labeled U',
        sns.distplot(self._Y_train.reshape(-1, ), hist=False, color=sns.xkcd_rgb["medium green"],
                     kde_kws={"shade": True, 'bw': 0.15}, ax=ax, label='Entire Train')  # label='Train',
        sns.distplot(self._Y_test.reshape(-1, ), hist=False, color=sns.xkcd_rgb["pale red"],
                     kde_kws={"shade": True, 'bw': 0.15}, ax=ax, label='Test')  # label='Test',
        if sum(self.batch_size) != 1:
            sns.distplot(ytr_last_batch.reshape(-1, ), hist=False, color=sns.xkcd_rgb["light purple"],
                     kde_kws={"shade": True, 'bw': 0.15}, ax=ax, label='Last Batch')  # label='Last Batch',
        # labels
        ax.set_ylabel(r'$\pi(Labels)$')
        ax.set_xlabel(r'$Labels$')

        # font size
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(14)

        # full except last batch
        fig2 = plt.figure()
        ax2 = sns.distplot(self._Y_pred.reshape(-1, ), hist=False, color=sns.xkcd_rgb["tangerine"],
                           kde_kws={"shade": True, 'bw': 0.15}, label='Predicted U')  # label='Predicted U',
        if Y is not None:
            sns.distplot(Y.reshape(-1, ), hist=False, color=sns.xkcd_rgb["denim blue"],
                         kde_kws={"shade": True, 'bw': 0.15}, ax=ax2, label='Labeled U')  # label='Labeled U',
        sns.distplot(self._Y_train.reshape(-1, ), hist=False, color=sns.xkcd_rgb["medium green"],
                     kde_kws={"shade": True, 'bw': 0.15}, ax=ax2, label='Entire Train')  # label='Train',
        sns.distplot(self._Y_test.reshape(-1, ), hist=False, color=sns.xkcd_rgb["pale red"],
                     kde_kws={"shade": True, 'bw': 0.15}, ax=ax2, label='Test')  # label='Test',

        # labels
        ax2.set_ylabel(r'$\pi(Labels)$')
        ax2.set_xlabel(r'$Labels$')

        # font size
        for item in ([ax2.title, ax2.xaxis.label, ax2.yaxis.label] +
                     ax2.get_xticklabels() + ax2.get_yticklabels()):
            item.set_fontsize(14)

        # full except Us
        fig3 = plt.figure()
        ax3 = sns.distplot(self._Y_train.reshape(-1, ), hist=False, color=sns.xkcd_rgb["medium green"],
                     kde_kws={"shade": True, 'bw': 0.15}, label='Entire Train')  # label='Train',
        sns.distplot(self._Y_test.reshape(-1, ), hist=False, color=sns.xkcd_rgb["pale red"],
                     kde_kws={"shade": True, 'bw': 0.15}, ax=ax3, label='Test')  # label='Test',
        if sum(self.batch_size) != 1:
            sns.distplot(ytr_last_batch.reshape(-1, ), hist=False, color=sns.xkcd_rgb["light purple"],
                     kde_kws={"shade": True, 'bw': 0.15}, ax=ax3, label='Last Batch')  # label='Last Batch',
        # labels
        ax3.set_ylabel(r'$\pi(Labels)$')
        ax3.set_xlabel(r'$Labels$')

        # font size
        for item in ([ax3.title, ax3.xaxis.label, ax3.yaxis.label] +
                     ax3.get_xticklabels() + ax3.get_yticklabels()):
            item.set_fontsize(14)

        return (fig1, fig2, fig3)

    def _visualize_learning_curve(self):
        """plot #3 : The learning curve of results
        """

        import matplotlib
        matplotlib.use('Agg')
        import seaborn as sns
        import matplotlib.pyplot as plt

        # data preparation
        algorithm = ['EMC'] * 3 * len(self._results)
        mae = list(self.results['mae'])
        mae += list(self.results['mae'] + self.results['mae_std'])
        mae += list(self.results['mae'] - self.results['mae_std'])
        size = list(self.results['num_training']) * 3
        if len(self._random_results) > 0 :
            pad_size = len(self._results) - len(self._random_results)
            algorithm += ['Random'] * 3 * len(self._results)
            mae += list(self.random_results['mae'])
            mae += list(self.random_results['mae'] + self.random_results['mae_std']) + [0] * pad_size
            mae += list(self.random_results['mae'] - self.random_results['mae_std']) + [0] * pad_size
            size += list(self.results['num_training']) * 3
        # dataframe
        dp = pd.DataFrame()
        dp['Algorithm'] = algorithm
        dp['Mean Absolute Error'] = mae
        dp['Training Size'] = size

        # figures
        sns.set_style('whitegrid')
        fig = plt.figure()
        ax = sns.lineplot(x='Training Size',
                          y='Mean Absolute Error',
                          style='Algorithm',
                          hue='Algorithm',
                          markers={'EMC': 'o', 'Random': 's'},
                          palette={'EMC': sns.xkcd_rgb["denim blue"],
                                   'Random': sns.xkcd_rgb["pale red"]},
                          data=dp,
                          )

        # font size
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(14)

        return fig