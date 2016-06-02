import warnings
import pandas as pd

class sklearn_Base(object):
    """
    Do not instantiate this class
    """
    def __init__(self, Base, parameters, iblock, SuperFunction):
        self.Base = Base
        self.parameters = parameters
        self.iblock = iblock
        self.SuperFunction = SuperFunction

    def run(self):
        self.legal_IO()
        self.receive()
        self.fit()
        self.send()

    def receive(self):
        recv = [edge for edge in self.Base.graph if edge[2] == self.iblock]
        self.Base.graph = [edge for edge in self.Base.graph if edge[2] != self.iblock]
        # check received tokens
        count = [0] * len(self.legal_inputs)
        for edge in recv:
            if edge[3] in self.legal_inputs:
                ind = self.legal_inputs.index(edge[2])
                count[ind] += 1
                if count[ind] > 1:
                    msg = '@Task #%i(%s): only one input per each available input can be received.' % (
                        self.iblock + 1, self.SuperFunction)
                    raise IOError(msg)
            else:
                msg = "@Task #%i(%s): received a non valid input token '%s', sent by function #%i" % (
                    self.iblock + 1, self.SuperFunction, edge[3], edge[0] + 1)
                raise IOError(msg)
        for edge in recv:
            key = edge[0:2]
            if key in self.Base.send:
                value = self.Base.send[key][0]
                self.legal_inputs[edge[3]] = value
                self.Base.send[key][1] -= 1
                if self.Base.send[key][1] == 0:
                    del self.Base.send[key]
            else:
                msg = '@Task #%i(%s): broken pipe in edge %s - nothing has been sent' % (
                    self.iblock + 1, self.SuperFunction, str(edge))
                raise IOError(msg)
        return self.legal_inputs

    def send(self):
        send = [edge for edge in self.Base.graph if edge[0] == self.iblock]
        for edge in send:
            key = edge[0:1]
            if key in self.Base.send:
                self.Base.send[key][1] += 1
            else:
                self.Base.send[key] = [self.legal_outputs[edge[1]], 1]

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
            warnings.warn("empty dataframe: all columns have been removed", Warning)
        else:
            stats = transformer.statistics_
            nan_ind = [i for i, val in enumerate(stats) if np.isnan(val)]
            df_columns = list_del_indices(df_columns, nan_ind)
            df = pd.DataFrame(df, columns=df_columns)
        return df

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
            warnings.warn("@Task #%i(%s): empty dataframe - all columns have been removed" % (
                self.iblock + 1, self.SuperFunction), Warning)
        if df.shape[1] == len(df_columns):
            df = pd.DataFrame(df, columns=df_columns)
        else:
            df = pd.DataFrame(df)
            warnings.warn(
                "@Task #%i(%s): headers untrackable - number of columns before and after transform doesn't match" % (
                    self.iblock + 1, self.SuperFunction), Warning)
        return df

    def selector_dataframe(self, transformer, df, tf):
        """ keep track of features (columns) that can be removed or changed in the
            VarianceThreshold by transforming data back to pandas dataframe structure.
            This happens based on the "get_support" method of selector.

        Parameters
        ----------
        imputer: sklearn VarianceThreshold class
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
            retained_features_ind = sel.get_support(True)
            df_columns = [df_columns[i] for i in retained_features_ind]
            df = pd.DataFrame(df, columns=df_columns)
            return df

#####################################################################

class PolynomialFeatures(sklearn_Base,Preprocessor):
    def legal_IO(self):
        self.legal_inputs = {'df': None}
        self.legal_outputs = {'api':None, 'df':None}
        self.Base.requirements.append('scikit_learn', 'pandas')

    def fit(self):
        from sklearn.preprocessing import PolynomialFeatures
        # check inputs
        if self.legal_inputs['df'] == None:
            msg = '@Task #%i(%s): input data frame is required'%(self.iblock,self.SuperFunction)
            raise IOError(msg)
        try:
            model = PolynomialFeatures(**self.parameters)
        except Exception as err:
            msg = '@Task #%i(%s): '%(self.iblock+1, self.SuperFunction) + type(err).__name__ + ': '+ err.message
            raise TypeError(msg)
        order = [edge[1] for edge in self.Base.graph if edge[0]==self.iblock]
        for token in order:
            if token == 'api':
                self.legal_outputs[token] = model
            elif token == 'df':
                self.legal_outputs[token] = self.Transformer_ManipulateHeader(model, self.legal_inputs['df'])
            else:
                msg = "@Task #%i(%s): non valid output token '%s'" % (self.iblock, self.SuperFunction, token)
                raise NameError(msg)

#####################################################################

class Imputer(sklearn_Base,Preprocessor):
    def legal_IO(self):
        self.legal_inputs = {'df': None}
        self.legal_outputs = {'api':None, 'df':None}
        self.Base.requirements.append('scikit_learn')

    def fit(self):
        from sklearn.preprocessing import Imputer
        try:
            model = Imputer(**self.parameters)
        except Exception as err:
            msg = '@Task #%i(%s): '%(self.iblock+1, self.SuperFunction) + type(err).__name__ + ': '+ err.message
            raise TypeError(msg)
        order = [edge[1] for edge in self.Base.graph if edge[0]==self.iblock]
        for token in order:
            if token == 'api':
                self.legal_outputs[token] = model
            elif token == 'df':
                self.legal_outputs[token] = self.Imputer_ManipulateHeader(model, self.legal_inputs['df'])
            else:
                msg = "@Task #%i(%s): non valid output token '%s'" % (self.iblock+1, self.SuperFunction, token)
                raise NameError(msg)

class StandardScaler(sklearn_Base,Preprocessor):
    def legal_IO(self):
        self.legal_inputs = {'df': None}
        self.legal_outputs = {'api':None, 'df':None}
        self.Base.requirements.append('scikit_learn')

    def fit(self):
        from sklearn.preprocessing import StandardScaler
        try:
            model = StandardScaler(**self.parameters)
        except Exception as err:
            msg = '@Task #%i(%s): '%(self.iblock+1, self.SuperFunction) + type(err).__name__ + ': '+ err.message
            raise TypeError(msg)
        order = [edge[1] for edge in self.Base.graph if edge[0]==self.iblock]
        for token in order:
            if token == 'api':
                self.legal_outputs[token] = model
            elif token == 'df':
                self.legal_outputs[token] = self.Transformer_ManipulateHeader(model, self.legal_inputs['df'])
            else:
                msg = "@Task #%i(%s): non valid output token '%s'" % (self.iblock, self.SuperFunction, token)
                raise NameError(msg)

class MinMaxScaler(sklearn_Base,Preprocessor):
    def legal_IO(self):
        self.legal_inputs = {'df': None}
        self.legal_outputs = {'api':None, 'df':None}
        self.Base.requirements.append('scikit_learn')

    def fit(self):
        from sklearn.preprocessing import MinMaxScaler
        try:
            model = MinMaxScaler(**self.parameters)
        except Exception as err:
            msg = '@Task #%i(%s): '%(self.iblock+1, self.SuperFunction) + type(err).__name__ + ': '+ err.message
            raise TypeError(msg)
        order = [edge[1] for edge in self.Base.graph if edge[0]==self.iblock]
        for token in order:
            if token == 'api':
                self.legal_outputs[token] = model
            elif token == 'df':
                self.legal_outputs[token] = self.Transformer_ManipulateHeader(model, self.legal_inputs['df'])
            else:
                msg = "@Task #%i(%s): non valid output token '%s'" % (self.iblock+1, self.SuperFunction, token)
                raise NameError(msg)

class MaxAbsScaler(sklearn_Base,Preprocessor):
    def legal_IO(self):
        self.legal_inputs = {'df': None}
        self.legal_outputs = {'api':None, 'df':None}
        self.Base.requirements.append('scikit_learn')

    def fit(self):
        from sklearn.preprocessing import MaxAbsScaler
        try:
            model = MaxAbsScaler(**self.parameters)
        except Exception as err:
            msg = '@Task #%i(%s): '%(self.iblock+1, self.SuperFunction) + type(err).__name__ + ': '+ err.message
            raise TypeError(msg)
        order = [edge[1] for edge in self.Base.graph if edge[0]==self.iblock]
        for token in order:
            if token == 'api':
                self.legal_outputs[token] = model
            elif token == 'df':
                self.legal_outputs[token] = self.Transformer_ManipulateHeader(model, self.legal_inputs['df'])
            else:
                msg = "@Task #%i(%s): non valid output token '%s'" % (self.iblock+1, self.SuperFunction, token)
                raise NameError(msg)

class RobustScaler(sklearn_Base,Preprocessor):
    def legal_IO(self):
        self.legal_inputs = {'df': None}
        self.legal_outputs = {'api':None, 'df':None}
        self.Base.requirements.append('scikit_learn')

    def fit(self):
        from sklearn.preprocessing import RobustScaler
        try:
            model = RobustScaler(**self.parameters)
        except Exception as err:
            msg = '@Task #%i(%s): '%(self.iblock+1, self.SuperFunction) + type(err).__name__ + ': '+ err.message
            raise TypeError(msg)
        order = [edge[1] for edge in self.Base.graph if edge[0]==self.iblock]
        for token in order:
            if token == 'api':
                self.legal_outputs[token] = model
            elif token == 'df':
                self.legal_outputs[token] = self.Transformer_ManipulateHeader(model, self.legal_inputs['df'])
            else:
                msg = "@Task #%i(%s): non valid output token '%s'" % (self.iblock+1, self.SuperFunction, token)
                raise NameError(msg)

class Normalizer(sklearn_Base,Preprocessor):
    def legal_IO(self):
        self.legal_inputs = {'df': None}
        self.legal_outputs = {'api':None, 'df':None}
        self.Base.requirements.append('scikit_learn')

    def fit(self):
        from sklearn.preprocessing import Normalizer
        try:
            model = Normalizer(**self.parameters)
        except Exception as err:
            msg = '@Task #%i(%s): '%(self.iblock+1, self.SuperFunction) + type(err).__name__ + ': '+ err.message
            raise TypeError(msg)
        order = [edge[1] for edge in self.Base.graph if edge[0]==self.iblock]
        for token in order:
            if token == 'api':
                self.legal_outputs[token] = model
            elif token == 'df':
                self.legal_outputs[token] = self.Transformer_ManipulateHeader(model, self.legal_inputs['df'])
            else:
                msg = "@Task #%i(%s): non valid output token '%s'" % (self.iblock+1, self.SuperFunction, token)
                raise NameError(msg)

class Binarizer(sklearn_Base,Preprocessor):
    def legal_IO(self):
        self.legal_inputs = {'df': None}
        self.legal_outputs = {'api':None, 'df':None}
        self.Base.requirements.append('scikit_learn')

    def fit(self):
        from sklearn.preprocessing import Binarizer
        try:
            model = Binarizer(**self.parameters)
        except Exception as err:
            msg = '@Task #%i(%s): '%(self.iblock+1, self.SuperFunction) + type(err).__name__ + ': '+ err.message
            raise TypeError(msg)
        order = [edge[1] for edge in self.Base.graph if edge[0]==self.iblock]
        for token in order:
            if token == 'api':
                self.legal_outputs[token] = model
            elif token == 'df':
                self.legal_outputs[token] = self.Transformer_ManipulateHeader(model, self.legal_inputs['df'])
            else:
                msg = "@Task #%i(%s): non valid output token '%s'" % (self.iblock+1, self.SuperFunction, token)
                raise NameError(msg)

class OneHotEncoder(sklearn_Base,Preprocessor):
    def legal_IO(self):
        self.legal_inputs = {'df': None}
        self.legal_outputs = {'api':None, 'df':None}
        self.Base.requirements.append('scikit_learn')

    def fit(self):
        from sklearn.preprocessing import OneHotEncoder
        try:
            model = OneHotEncoder(**self.parameters)
        except Exception as err:
            msg = '@Task #%i(%s): '%(self.iblock+1, self.SuperFunction) + type(err).__name__ + ': '+ err.message
            raise TypeError(msg)
        order = [edge[1] for edge in self.Base.graph if edge[0]==self.iblock]
        for token in order:
            if token == 'api':
                self.legal_outputs[token] = model
            elif token == 'df':
                self.legal_outputs[token] = self.Transformer_ManipulateHeader(model, self.legal_inputs['df'])
            else:
                msg = "@Task #%i(%s): non valid output token '%s'" % (self.iblock+1, self.SuperFunction, token)
                raise NameError(msg)

#####################################################################
