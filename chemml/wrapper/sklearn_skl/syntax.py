import pandas as pd
import numpy as np
import warnings
import inspect

from chemml.utils.utilities import list_del_indices

class Preprocessor(object):
    def imputer(self, transformer, df, method):
        """ keep track of features (columns) that can be removed or changed in the
            Imputer by transforming data back to pandas dataframe structure. This happens based on
            the "statistics_" attribute of Imputer.

            this function also distinguish different methods of Imputer.
        Parameters
        ----------
        transformer: sklearn Imputer class
             The class with adjusted parameters.

        df: Pandas dataframe
            The dataframe that imputer is going to deal with.

        method: string
            sklearn methods for imputer: 'fit_transform' and 'transform'

        Returns
        -------
        transformed data frame
        fitted imputer class
        """

        df_columns = list(df.columns)
        if method == 'fit_transform':
            df = transformer.fit_transform(df)
        elif method == 'transform':
            df = transformer.transform(df)
        else:
            msg = "@Task #%i(%s): The passed method is not valid. It can be any of fit_transform, transform." % (self.iblock, self.Task)
            raise NameError(msg)
        if df.shape[1] == 0:
            warnings.warn("@Task #%i(%s): empty dataframe: all columns have been removed" \
                          %(self.iblock + 1, self.Task), Warning)
        else:
            stats = transformer.statistics_
            nan_ind = [i for i, val in enumerate(stats) if np.isnan(val)]
            df_columns = list_del_indices(df_columns, nan_ind)
            df = pd.DataFrame(df, columns=df_columns)
        return transformer, df

    def fit_transform_inverse(self, api, df, method, header=True):
        """ keep track of features (columns) that can be removed or changed in the
            preprocessor by transforming data back to pandas dataframe.

            this function also distinguish different methods of APIs.

        Parameters
        ----------
        api: an instance of sklearn class
            The class with adjusted parameters.

        df: Pandas dataframe
            The dataframe that api is going to deal with

        method: string
            sklearn methods for the api: 'fit_transform', 'transform', 'inverse_transform'

        Returns
        -------
        transformed data frame
        fitted api class

        """

        df_columns = list(df.columns)

        if method == 'fit_transform':
            df = api.fit_transform(df)
        elif method == 'transform':
            df = api.transform(df)
        elif method == 'inverse_transform':
            df = api.inverse_transform(df)
        if header:
            if df.shape[1] == 0:
                warnings.warn("@Task #%i(%s): empty dataframe - all columns have been removed"
                              % (self.iblock + 1, self.Task), Warning)
            if df.shape[1] == len(df_columns):
                df = pd.DataFrame(df, columns=df_columns)
            else:
                df = pd.DataFrame(df)
                warnings.warn("@Task #%i(%s): headers are untrackable - number of columns before and after transform doesn't match" \
                    %(self.iblock + 1, self.Task), Warning)
        return api, df

    #unused
    def selector_dataframe(self, transformer, df, tf):
        """ keep track of features (columns) that can be removed or changed in the
            VarianceThreshold by transforming data back to pandas dataframe structure.
            This happens based on the "get_support" method of selector.

        Parameters
        ----------
        transformer: sklearn VarianceThreshold class
             The class with adjusted parameters.

        df: Pandas dataframe
            data frame

        tf: Pandas dataframe
            target frame

        Returns
        -------
        transformed data frame
        fitted imputer class
        """
        df_columns = list(df.columns)
        df = transformer.fit_transform(df, tf)
        if df.shape[1] == 0:
            warnings.warn("empty dataframe: all columns have been removed", Warning)
            return transformer, df
        else:
            retained_features_ind = transformer.get_support(True)
            df_columns = [df_columns[i] for i in retained_features_ind]
            df = pd.DataFrame(df, columns=df_columns)
            return df

class Regressor(object):
    def sklearn_training(self,model,X,y):
        model.fit(X, y)
        r2score_training = model.score(X, y)
        return model, r2score_training

    def train(self, model, X, Y):
        if model[1][2] == 'sklearn':
            model[0].fit(X, Y)
            model_trained = model[0]
            r2score_training = model[0].score(X, Y)
        elif model[1][2] == 'chemml' and model[1][3] == 'NN_PSGD':
            from chemml.nn import nn_psgd
            ev_inds = np.random.choice(np.arange(X.shape[0]),int(0.2*X.shape[0])+1,replace=False)
            tr_inds = np.delete(np.arange(X.shape[0]),ev_inds)
            x_tr = X.loc[tr_inds,:]
            x_ev = X.loc[ev_inds,:]
            y_tr = Y.loc[tr_inds,:]
            y_ev = Y.loc[ev_inds,:]
            model_trained = nn_psgd.train(x_tr, x_ev, y_tr, y_ev, **model[0])
            from sklearn.metrics import r2_score
            r2score_training = r2_score(y_true=Y, y_pred=nn_psgd.output(X,model_trained), multioutput='raw_values')
        return model_trained,r2score_training


    def predict(self, model, X):
        if model[1][2] == 'sklearn':
            Y_pred = pd.DataFrame(model[0].predict(X))
        elif model[1][2] == 'chemml' and model[1][3] == 'NN_PSGD':
            from chemml.nn import nn_psgd
            Y_pred = nn_psgd.output(X.values, model[0])
        return Y_pred

class Evaluator(object):
    def _reg_evaluation_params(self):
        self.params = {}
        self.evaluator = {}
        if 'r2_score' in self.parameters and self.parameters['r2_score'] == True:
            metric = 'r2_score'
            self.params[metric] = {}
            if 'r2_sample_weight' in self.parameters:
                self.params[metric]['sample_weight'] = self.parameters['r2_sample_weight']
            if 'r2_multioutput' in self.parameters:
                self.params[metric]['multioutput'] = self.parameters['r2_multioutput']

            # import module and make APIs
            from sklearn.metrics import r2_score
            self.evaluator[metric] = lambda Y, Y_pred: r2_score(y_true=Y, y_pred=Y_pred, **self.params[metric])

        if 'mean_absolute_error' in self.parameters and self.parameters['mean_absolute_error'] == True:
            metric = 'mean_absolute_error'
            self.params[metric] = {}
            if 'mae_sample_weight' in self.parameters:
                self.params[metric]['sample_weight'] = self.parameters['mae_sample_weight']
            if 'mae_multioutput' in self.parameters:
                self.params[metric]['multioutput'] = self.parameters['mae_multioutput']

            # import module and make APIs
            from sklearn.metrics import mean_absolute_error
            self.evaluator[metric] = lambda Y, Y_pred: mean_absolute_error(y_true=Y, y_pred=Y_pred, **self.params[metric])

        if 'median_absolute_error' in self.parameters and self.parameters['median_absolute_error'] == True:
            metric = 'median_absolute_error'
            self.params[metric] = {}

            # import module and make APIs
            from sklearn.metrics import median_absolute_error
            self.evaluator[metric] = lambda Y, Y_pred: median_absolute_error(y_true=Y, y_pred=Y_pred)#, **self.params[metric])

        if 'mean_squared_error' in self.parameters and self.parameters['mean_squared_error'] == True:
            metric = 'mean_squared_error'
            self.params[metric] = {}
            if 'mse_sample_weight' in self.parameters:
                self.params[metric]['sample_weight'] = self.parameters['mse_sample_weight']
            if 'mse_multioutput' in self.parameters:
                self.params[metric]['multioutput'] = self.parameters['mse_multioutput']

            # import module and make APIs
            from sklearn.metrics import mean_squared_error
            self.evaluator[metric] = lambda Y, Y_pred: mean_squared_error(y_true=Y, y_pred=Y_pred, **self.params[metric])

        if 'root_mean_squared_error' in self.parameters and self.parameters['root_mean_squared_error'] == True:
            metric = 'root_mean_squared_error'
            self.params[metric] = {}
            if 'rmse_sample_weight' in self.parameters:
                self.params[metric]['sample_weight'] = self.parameters['rmse_sample_weight']
            if 'rmse_multioutput' in self.parameters:
                self.params[metric]['multioutput'] = self.parameters['rmse_multioutput']

            # import module and make APIs
            from sklearn.metrics import mean_squared_error
            self.evaluator[metric] = lambda Y, Y_pred: np.sqrt(mean_squared_error(y_true=Y, y_pred=Y_pred, **self.params[metric]))

        if 'explained_variance_score' in self.parameters and self.parameters['explained_variance_score'] == True:
            metric = 'explained_variance_score'
            self.params[metric] = {}
            if 'ev_sample_weight' in self.parameters:
                self.params[metric]['sample_weight'] = self.parameters['ev_sample_weight']
            if 'ev_multioutput' in self.parameters:
                self.params[metric]['multioutput'] = self.parameters['ev_multioutput']

            # import module and make APIs
            from sklearn.metrics import explained_variance_score
            self.evaluator[metric] = lambda Y, Y_pred: explained_variance_score(y_true=Y, y_pred=Y_pred, **self.params[metric])

    def _reg_evaluate(self, Y, Y_pred, evaluator):
        if Y.shape[1]>1:
            msg = "@Task #%i(%s): be aware of 'multioutput' behavior of selected metrics" \
                          %(self.iblock + 1, self.Task)
            warnings.warn(msg, Warning)
        self.results = {}
        for metric in evaluator:
            try:
                self.results[metric] = [evaluator[metric](Y, Y_pred)]
            except Exception as err:
                msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(
                    err).__name__ + ': ' + err.message
                raise TypeError(msg)


def Fit(fn):
    """ Fit with a wrapper """
    def wrapper(self):
        self.paramFROMinput()
        if 'track_header' in self.parameters:
            self.header = self.parameters.pop('track_header')
        else:
            self.header = True
        if 'func_method' in self.parameters:
            self.method = self.parameters.pop('func_method')
        else:
            self.method = None
        available_methods = self.metadata.WParameters.func_method.options
        if self.method not in available_methods:
            msg = "@Task #%i(%s): The method '%s' is not available for the function '%s'." % (
                self.iblock, self.Task,self.method,self.Function)
            raise NameError(msg)
        else:
            if self.method == None:
                api = self.import_sklearn()
                self.set_value('api', api)
                fn(self)
            elif self.method == 'fit_transform':
                api = self.import_sklearn()
                self.required('df', req=True)
                df = self.inputs['df'].value
                df = api.fit_transform(df)
                self.set_value('api', api)
                self.set_value('df', df)
                fn(self)
            elif self.method == 'transform':
                self.required('df', req=True)
                self.required('api', req=True)
                df = self.inputs['df'].value
                api = self.inputs['api'].value
                df = api.transform(df)
                self.set_value('api', api)
                self.set_value('df', df)
                fn(self)
            elif self.method == 'inverse_transform':
                self.required('df', req=True)
                self.required('api', req=True)
                df = self.inputs['df'].value
                api = self.inputs['api'].value
                df = api.inverse_transform(df)
                self.set_value('api', api)
                self.set_value('df', df)
            elif self.method == 'fit':
                api = self.import_sklearn()
                self.required('dfx', req=True)
                self.required('dfy', req=True)
                dfx = self.inputs['dfx'].value
                dfy = self.inputs['dfy'].value
                api.fit(dfx,dfy)
                dfy_predict = api.predict(dfx)
                self.set_value('api', api)
                self.set_value('dfy_predict', dfy_predict)
                fn(self)
            elif self.method == 'predict':
                self.required('dfx', req=True)
                self.required('api', req=True)
                dfx = self.inputs['dfx'].value
                api = self.inputs['api'].value
                dfy_predict = api.predict(dfx)
                self.set_value('api', api)
                self.set_value('dfy_predict', dfy_predict)
                fn(self)

        self.Send()

        # delete all inputs
        del self.inputs