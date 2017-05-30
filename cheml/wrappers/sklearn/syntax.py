import pandas as pd
import numpy as np
import warnings

from ...utils.utilities import list_del_indices

class Preprocessor(object):
    def Imputer_ManipulateHeader(self, transformer, df):
        """ keep track of features (columns) that can be removed or changed in the
            Imputer by transforming data back to pandas dataframe structure. This happens based on
            the "statistics_" attribute of Imputer.

        Parameters
        ----------
        transformer: sklearn Imputer class
             The class with adjusted parameters.

        df: Pandas dataframe
            The dataframe that imputer is going to deal with.

        Returns
        -------
        transformed data frame
        fitted imputer class
        """

        df_columns = list(df.columns)
        df = transformer.fit_transform(df)
        if df.shape[1] == 0:
            warnings.warn("@Task #%i(%s): empty dataframe: all columns have been removed" \
                          %(self.iblock + 1, self.SuperFunction), Warning)
        else:
            stats = transformer.statistics_
            nan_ind = [i for i, val in enumerate(stats) if np.isnan(val)]
            df_columns = list_del_indices(df_columns, nan_ind)
            df = pd.DataFrame(df, columns=df_columns)
        return transformer, df

    def scaler(self, transformer, df, method):
        """ keep track of features (columns) that can be removed or changed in the
            Scaler by transforming data back to pandas dataframe structure.

        Parameters
        ----------
        transformer: an instance of sklearn Scaler class
            The class with adjusted parameters.

        df: Pandas dataframe
            The dataframe that scaler is going to deal with.

        method: string
            sklearn methods for scaler: 'fit_transform', 'transform', inverse_transform'

        Returns
        -------
        transformed data frame
        fitted scaler class

        """

        df_columns = list(df.columns)
        if method == 'fit_transform':
            df = transformer.fit_transform(df)
        elif method == 'transform':
            df = transformer.transform(df)
        elif method == 'inverse':
            df = transformer.inverse_transform(df)
        else:
            msg = "@Task #%i(%s): The passed method is not valid. It can be any of fit_transform, transform or inverse_transform." % (self.iblock, self.SuperFunction)
            raise NameError(msg)

        if df.shape[1] == 0:
            warnings.warn("@Task #%i(%s): empty dataframe - all columns have been removed" \
                          % (self.iblock + 1, self.SuperFunction), Warning)
        if df.shape[1] == len(df_columns):
            df = pd.DataFrame(df, columns=df_columns)
        else:
            df = pd.DataFrame(df)
            warnings.warn("@Task #%i(%s): headers untrackable - number of columns before and after transform doesn't match" \
                %(self.iblock + 1, self.SuperFunction), Warning)
        return transformer, df


    def Transformer_ManipulateHeader(self, transformer, df):
        """ keep track of features (columns) that can be removed or changed in the
            Scaler by transforming data back to pandas dataframe structure.

        Parameters
        ----------
        scaler: sklearn Scaler class
            The class with adjusted parameters.

        df: Pandas dataframe
            The dataframe that scaler is going to deal with.

        Returns
        -------
        transformed data frame
        fitted scaler class

        """
        df_columns = list(df.columns)
        df = transformer.fit_transform(df)
        if df.shape[1] == 0:
            warnings.warn("@Task #%i(%s): empty dataframe - all columns have been removed" \
                          % (self.iblock + 1, self.SuperFunction), Warning)
        if df.shape[1] == len(df_columns):
            df = pd.DataFrame(df, columns=df_columns)
        else:
            df = pd.DataFrame(df)
            warnings.warn("@Task #%i(%s): headers untrackable - number of columns before and after transform doesn't match" \
                %(self.iblock + 1, self.SuperFunction), Warning)
        return transformer, df

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
        elif model[1][2] == 'cheml' and model[1][3] == 'NN_PSGD':
            from cheml.nn import nn_psgd
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
        elif model[1][2] == 'cheml' and model[1][3] == 'NN_PSGD':
            from cheml.nn import nn_psgd
            Y_pred = nn_psgd.output(X.values, model[0])
        return Y_pred

class Evaluator(object):
    def _evaluation_params(self):
        self.params = {}
        self.results = {}
        if 'r2_score' in self.parameters and self.parameters['r2_score'] == True:
            self.results['r2_score'] = []
            self.params['r2_score'] = {}
            if 'r2_sample_weight' in self.parameters:
                self.params['r2_score']['sample_weight'] = self.parameters['r2_sample_weight']
            if 'r2_multioutput' in self.parameters:
                self.params['r2_score']['multioutput'] = self.parameters['r2_multioutput']
        if 'mean_absolute_error' in self.parameters and self.parameters['mean_absolute_error'] == True:
            self.results['mean_absolute_error'] = []
            self.params['mean_absolute_error'] = {}
            if 'mae_sample_weight' in self.parameters:
                self.params['mean_absolute_error']['sample_weight'] = self.parameters['mae_sample_weight']
            if 'mae_multioutput' in self.parameters:
                self.params['mean_absolute_error']['multioutput'] = self.parameters['mae_multioutput']
        if 'median_absolute_error' in self.parameters and self.parameters['median_absolute_error'] == True:
            self.results['median_absolute_error'] = []
            self.params['median_absolute_error'] = {}
        if 'mean_squared_error' in self.parameters and self.parameters['mean_squared_error'] == True:
            self.results['mean_squared_error'] = []
            self.params['mean_squared_error'] = {}
            if 'mse_sample_weight' in self.parameters:
                self.params['mean_squared_error']['sample_weight'] = self.parameters['mse_sample_weight']
            if 'mse_multioutput' in self.parameters:
                self.params['mean_squared_error']['multioutput'] = self.parameters['mse_multioutput']
        if 'root_mean_squared_error' in self.parameters and self.parameters['root_mean_squared_error'] == True:
            self.results['root_mean_squared_error'] = []
            self.params['root_mean_squared_error'] = {}
            if 'rmse_sample_weight' in self.parameters:
                self.params['root_mean_squared_error']['sample_weight'] = self.parameters['rmse_sample_weight']
            if 'rmse_multioutput' in self.parameters:
                self.params['root_mean_squared_error']['multioutput'] = self.parameters['rmse_multioutput']
        if 'explained_variance_score' in self.parameters and self.parameters['explained_variance_score'] == True:
            self.results['explained_variance_score'] = []
            self.params['explained_variance_score'] = {}
            if 'ev_sample_weight' in self.parameters:
                self.params['explained_variance_score']['sample_weight'] = self.parameters['ev_sample_weight']
            if 'ev_multioutput' in self.parameters:
                self.params['explained_variance_score']['multioutput'] = self.parameters['ev_multioutput']

    def evaluate(self, Y, Y_pred):
        # if Y.shape[1]>1:
        #     msg = "@Task #%i(%s): multiple output values - be careful with the multioutput parameter in the selected metric" \
        #                   %(self.iblock + 1, self.SuperFunction)
        #     warnings.warn(msg, Warning)
        for metric in self.results:
            if metric == 'r2_score':
                from sklearn.metrics import r2_score
                try:
                    self.results[metric].append(r2_score(y_true=Y, y_pred=Y_pred, **self.params[metric]))
                except Exception as err:
                    msg = '@Task #%i(%s): '%(self.iblock+1, self.SuperFunction) + type(err).__name__ + ': '+ err.message
                    raise TypeError(msg)
            elif metric == 'mean_absolute_error':
                from sklearn.metrics import mean_absolute_error
                try:
                    self.results[metric].append(mean_absolute_error(y_true=Y, y_pred=Y_pred, **self.params[metric]))
                except Exception as err:
                    msg = '@Task #%i(%s): '%(self.iblock+1, self.SuperFunction) + type(err).__name__ + ': '+ err.message
                    raise TypeError(msg)
            elif metric == 'median_absolute_error':
                from sklearn.metrics import median_absolute_error
                try:
                    self.results[metric].append(median_absolute_error(y_true=Y, y_pred=Y_pred))#, **self.params[metric])
                except Exception as err:
                    msg = '@Task #%i(%s): '%(self.iblock+1, self.SuperFunction) + type(err).__name__ + ': '+ err.message
                    raise TypeError(msg)
            elif metric == 'mean_squared_error':
                from sklearn.metrics import mean_squared_error
                try:
                    self.results[metric].append(mean_squared_error(y_true=Y, y_pred=Y_pred, **self.params[metric]))
                except Exception as err:
                    msg = '@Task #%i(%s): '%(self.iblock+1, self.SuperFunction) + type(err).__name__ + ': '+ err.message
                    raise TypeError(msg)
            elif metric == 'root_mean_squared_error':
                from sklearn.metrics import mean_squared_error
                try:
                    self.results[metric].append(np.sqrt(mean_squared_error(y_true=Y, y_pred=Y_pred, **self.params[metric])))
                except Exception as err:
                    msg = '@Task #%i(%s): '%(self.iblock+1, self.SuperFunction) + type(err).__name__ + ': '+ err.message
                    raise TypeError(msg)
            elif metric == 'explained_variance_score':
                from sklearn.metrics import explained_variance_score
                try:
                    self.results[metric].append(explained_variance_score(y_true=Y, y_pred=Y_pred, **self.params[metric]))
                except Exception as err:
                    msg = '@Task #%i(%s): '%(self.iblock+1, self.SuperFunction) + type(err).__name__ + ': '+ err.message
                    raise TypeError(msg)

