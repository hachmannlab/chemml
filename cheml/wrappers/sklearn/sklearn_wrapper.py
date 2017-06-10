import warnings
import pandas as pd
import numpy as np
import sklearn
import copy

from ..base import BASE, LIBRARY
from .syntax import Preprocessor, Regressor, Evaluator


##################################################################### 2 Prepare Data

# Data Representation

class PolynomialFeatures(BASE,LIBRARY,Preprocessor):
    def legal_IO(self):
        self.legal_inputs = {'df': None, 'api':None}
        self.legal_outputs = {'api':None, 'df':None}
        requirements = ['scikit_learn']
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def fit(self):
        # step1: check inputs
        df, df_info = self.input_check('df', req=True, py_type=pd.DataFrame)
        model, _ = self.input_check('api', req=False)

        # step2: assign inputs to parameters if necessary (param = @token)
        self.paramFROMinput()

        # step3: check the dimension of input data frame
        # df, _ = self.data_check('df', df, ndim=2, n0=None, n1=None, format_out='df')

        # step4: import module and make APIs
        method = self.parameters.pop('func_method')  # method = fit_transform, transform
        if isinstance(self.legal_inputs['api'], type(None)):
            if method == 'fit_transform':
                try:
                    from sklearn.preprocessing import PolynomialFeatures
                    model = PolynomialFeatures(**self.parameters)
                except Exception as err:
                    msg = '@Task #%i(%s): '%(self.iblock+1, self.SuperFunction) + type(err).__name__ + ': '+ err.message
                    raise TypeError(msg)
            else:
                msg = "@Task #%i(%s): pass an api to transform the input data, otherwise you need to fit_transform the data with proper parameters." % (self.iblock, self.SuperFunction)
                raise NameError(msg)

        # step5: process
        available_methods = ['fit_transform', 'transform']
        model, df = self.fit_transform_inverse(model, df, method, available_methods, header=False)

        # step6: send out
        order = [edge[1] for edge in self.Base.graph if edge[0]==self.iblock]
        for token in set(order):
            if token == 'api':
                self.Base.send[(self.iblock, token)] = [model, order.count(token),(self.iblock,token,self.Host,self.Function)]
            elif token == 'df':
                self.Base.send[(self.iblock, token)] = [df, order.count(token),(self.iblock,token,self.Host,self.Function)]
            else:
                msg = "@Task #%i(%s): non valid output token '%s'" % (self.iblock+1, self.SuperFunction, token)
                raise NameError(msg)

        # step7: delete all inputs from memory
        del self.legal_inputs


# Preprocessors

class Imputer(BASE,LIBRARY,Preprocessor):
    def legal_IO(self):
        self.legal_inputs = {'df': None, 'api':None}
        self.legal_outputs = {'api':None, 'df':None}
        requirements = ['scikit_learn']
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def fit(self):
        # step1: check inputs
        df, df_info = self.input_check('df', req=True, py_type=pd.DataFrame)
        model, _ = self.input_check('api', req=False)

        # step2: assign inputs to parameters if necessary (param = @token)
        self.paramFROMinput()

        # step3: check the dimension of input data frame
        # df, _ = self.data_check('df', df, ndim=2, n0=None, n1=None, format_out='df')

        # step4: import module and make APIs
        method = self.parameters.pop('func_method')  # method = fit_transform, transform or inverse_transform
        if isinstance(self.legal_inputs['api'], type(None)):
            if method == 'fit_transform':
                try:
                    from sklearn.preprocessing import Imputer
                    model = Imputer(**self.parameters)
                except Exception as err:
                    msg = '@Task #%i(%s): '%(self.iblock+1, self.SuperFunction) + type(err).__name__ + ': '+ err.message
                    raise TypeError(msg)
            else:
                msg = "@Task #%i(%s): pass an api to transform the input data, otherwise you need to fit_transform the data with proper parameters." % (self.iblock, self.SuperFunction)
                raise NameError(msg)

        # step5: process
        model, df = self.imputer(model, df, method)

        # step6: send out
        order = [edge[1] for edge in self.Base.graph if edge[0]==self.iblock]
        for token in set(order):
            if token == 'api':
                self.Base.send[(self.iblock, token)] = [model, order.count(token),(self.iblock,token,self.Host,self.Function)]
            elif token == 'df':
                self.Base.send[(self.iblock, token)] = [df, order.count(token),(self.iblock,token,self.Host,self.Function)]
            else:
                msg = "@Task #%i(%s): non valid output token '%s'" % (self.iblock+1, self.SuperFunction, token)
                raise NameError(msg)

        # step7: delete all inputs from memory
        del self.legal_inputs

class StandardScaler(BASE,LIBRARY,Preprocessor):
    def legal_IO(self):
        self.legal_inputs = {'df': None, 'api': None}
        self.legal_outputs = {'api':None, 'df':None}
        requirements = ['scikit_learn']
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def fit(self):
        # step1: check inputs
        df, df_info = self.input_check('df', req=True, py_type=pd.DataFrame)
        model, _ = self.input_check('api', req=False)

        # step2: assign inputs to parameters if necessary (param = @token)
        self.paramFROMinput()

        # step3: check the dimension of input data frame
        # df, _ = self.data_check('df', df, ndim=2, n0=None, n1=None, format_out='df')

        # step4: import module and make APIs
        method = self.parameters.pop('func_method')  # method = fit_transform, transform or inverse_transform
        if isinstance(self.legal_inputs['api'], type(None)):
            if method == 'fit_transform':
                try:
                    from sklearn.preprocessing import StandardScaler
                    model = StandardScaler(**self.parameters)
                except Exception as err:
                    msg = '@Task #%i(%s): '%(self.iblock+1, self.SuperFunction) + type(err).__name__ + ': '+ err.message
                    raise TypeError(msg)
            else:
                msg = "@Task #%i(%s): pass an api to transform or inverse_transform the input data, otherwise you need to fit_transform the data with proper parameters." % (self.iblock, self.SuperFunction)
                raise NameError(msg)

        # step5: process
        available_methods = ['fit_transform', 'transform', 'inverse_transform']
        model, df = self.fit_transform_inverse(model, df, method, available_methods, header=True)

        # step6: send out
        order = [edge[1] for edge in self.Base.graph if edge[0]==self.iblock]
        for token in set(order):
            if token == 'api':
                self.Base.send[(self.iblock, token)] = [model, order.count(token),(self.iblock,token,self.Host,self.Function)]
            elif token == 'df':
                self.Base.send[(self.iblock, token)] = [df, order.count(token),(self.iblock,token,self.Host,self.Function)]
            else:
                msg = "@Task #%i(%s): non valid output token '%s'" % (self.iblock, self.SuperFunction, token)
                raise NameError(msg)

        # step7: delete all inputs from memory
        del self.legal_inputs

class MinMaxScaler(BASE,LIBRARY,Preprocessor):
    def legal_IO(self):
        self.legal_inputs = {'df': None, 'api': None}
        self.legal_outputs = {'api':None, 'df':None}
        requirements = ['scikit_learn']
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def fit(self):
        # step1: check inputs
        df, df_info = self.input_check('df', req=True, py_type=pd.DataFrame)
        model, _ = self.input_check('api', req=False)

        # step2: assign inputs to parameters if necessary (param = @token)
        self.paramFROMinput()

        # step3: check the dimension of input data frame
        # df, _ = self.data_check('df', df, ndim=2, n0=None, n1=None, format_out='df')

        # step4: import module and make APIs
        method = self.parameters.pop('func_method')  # method = fit_transform, transform or inverse_transform
        if isinstance(self.legal_inputs['api'], type(None)):
            if method == 'fit_transform':
                try:
                    from sklearn.preprocessing import MinMaxScaler
                    model = MinMaxScaler(**self.parameters)
                except Exception as err:
                    msg = '@Task #%i(%s): '%(self.iblock+1, self.SuperFunction) + type(err).__name__ + ': '+ err.message
                    raise TypeError(msg)
            else:
                msg = "@Task #%i(%s): pass an api to transform or inverse_transform the input data, otherwise you need to fit_transform the data with proper parameters." % (self.iblock, self.SuperFunction)
                raise NameError(msg)

        # step5: process
        available_methods = ['fit_transform', 'transform', 'inverse_transform']
        model, df = self.fit_transform_inverse(model, df, method, available_methods, header=True)

        # step6: send out
        order = [edge[1] for edge in self.Base.graph if edge[0]==self.iblock]
        for token in set(order):
            if token == 'api':
                val = model
                self.Base.send[(self.iblock, token)] = [val, order.count(token),(self.iblock,token,self.Host,self.Function)]
            elif token == 'df':
                self.Base.send[(self.iblock, token)] = [df, order.count(token),(self.iblock,token,self.Host,self.Function)]
            else:
                msg = "@Task #%i(%s): non valid output token '%s'" % (self.iblock, self.SuperFunction, token)
                raise NameError(msg)

        # step7: delete all inputs from memory
        del self.legal_inputs

class MaxAbsScaler(BASE,LIBRARY,Preprocessor):
    def legal_IO(self):
        self.legal_inputs = {'df': None, 'api': None}
        self.legal_outputs = {'api':None, 'df':None}
        requirements = ['scikit_learn']
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def fit(self):
        # step1: check inputs
        df, df_info = self.input_check('df', req=True, py_type=pd.DataFrame)
        model, _ = self.input_check('api', req=False)

        # step2: assign inputs to parameters if necessary (param = @token)
        self.paramFROMinput()

        # step3: check the dimension of input data frame
        # df, _ = self.data_check('df', df, ndim=2, n0=None, n1=None, format_out='df')

        # step4: import module and make APIs
        method = self.parameters.pop('func_method')  # method = fit_transform, transform or inverse_transform
        if isinstance(self.legal_inputs['api'], type(None)):
            if method == 'fit_transform':
                try:
                    from sklearn.preprocessing import MaxAbsScaler
                    model = MaxAbsScaler(**self.parameters)
                except Exception as err:
                    msg = '@Task #%i(%s): '%(self.iblock+1, self.SuperFunction) + type(err).__name__ + ': '+ err.message
                    raise TypeError(msg)
            else:
                msg = "@Task #%i(%s): pass an api to transform or inverse_transform the input data, otherwise you need to fit_transform the data with proper parameters." % (self.iblock, self.SuperFunction)
                raise NameError(msg)

        # step5: process
        available_methods = ['fit_transform', 'transform', 'inverse_transform']
        model, df = self.fit_transform_inverse(model, df, method, available_methods, header=True)

        # step6: send out
        order = [edge[1] for edge in self.Base.graph if edge[0]==self.iblock]
        for token in set(order):
            if token == 'api':
                val = model
                self.Base.send[(self.iblock, token)] = [val, order.count(token),(self.iblock,token,self.Host,self.Function)]
            elif token == 'df':
                self.Base.send[(self.iblock, token)] = [df, order.count(token),(self.iblock,token,self.Host,self.Function)]
            else:
                msg = "@Task #%i(%s): non valid output token '%s'" % (self.iblock, self.SuperFunction, token)
                raise NameError(msg)

        # step7: delete all inputs from memory
        del self.legal_inputs

class RobustScaler(BASE,LIBRARY,Preprocessor):
    def legal_IO(self):
        self.legal_inputs = {'df': None, 'api': None}
        self.legal_outputs = {'api':None, 'df':None}
        requirements = ['scikit_learn']
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def fit(self):
        # step1: check inputs
        df, df_info = self.input_check('df', req=True, py_type=pd.DataFrame)
        model, _ = self.input_check('api', req=False)

        # step2: assign inputs to parameters if necessary (param = @token)
        self.paramFROMinput()

        # step3: check the dimension of input data frame
        # df, _ = self.data_check('df', df, ndim=2, n0=None, n1=None, format_out='df')

        # step4: import module and make APIs
        method = self.parameters.pop('func_method')  # method = fit_transform, transform or inverse_transform
        if isinstance(self.legal_inputs['api'], type(None)):
            if method == 'fit_transform':
                try:
                    from sklearn.preprocessing import RobustScaler
                    model = RobustScaler(**self.parameters)
                except Exception as err:
                    msg = '@Task #%i(%s): '%(self.iblock+1, self.SuperFunction) + type(err).__name__ + ': '+ err.message
                    raise TypeError(msg)
            else:
                msg = "@Task #%i(%s): pass an api to transform or inverse_transform the input data, otherwise you need to fit_transform the data with proper parameters." % (self.iblock, self.SuperFunction)
                raise NameError(msg)

        # step5: process
        available_methods = ['fit_transform', 'transform', 'inverse_transform']
        model, df = self.fit_transform_inverse(model, df, method, available_methods, header=True)

        # step6: send out
        order = [edge[1] for edge in self.Base.graph if edge[0]==self.iblock]
        for token in set(order):
            if token == 'api':
                val = model
                self.Base.send[(self.iblock, token)] = [val, order.count(token),(self.iblock,token,self.Host,self.Function)]
            elif token == 'df':
                self.Base.send[(self.iblock, token)] = [df, order.count(token),(self.iblock,token,self.Host,self.Function)]
            else:
                msg = "@Task #%i(%s): non valid output token '%s'" % (self.iblock, self.SuperFunction, token)
                raise NameError(msg)

        # step7: delete all inputs from memory
        del self.legal_inputs

class Normalizer(BASE,LIBRARY,Preprocessor):
    def legal_IO(self):
        self.legal_inputs = {'df': None, 'api':None}
        self.legal_outputs = {'api':None, 'df':None}
        requirements = ['scikit_learn']
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def fit(self):
        # step1: check inputs
        df, df_info = self.input_check('df', req=True, py_type=pd.DataFrame)
        model, _ = self.input_check('api', req=False)

        # step2: assign inputs to parameters if necessary (param = @token)
        self.paramFROMinput()

        # step3: check the dimension of input data frame
        # df, _ = self.data_check('df', df, ndim=2, n0=None, n1=None, format_out='df')

        # step4: import module and make APIs
        method = self.parameters.pop('func_method')  # method = fit_transform, transform or inverse_transform
        if isinstance(self.legal_inputs['api'], type(None)):
            if method == 'fit_transform':
                try:
                    from sklearn.preprocessing import Normalizer
                    model = Normalizer(**self.parameters)
                except Exception as err:
                    msg = '@Task #%i(%s): '%(self.iblock+1, self.SuperFunction) + type(err).__name__ + ': '+ err.message
                    raise TypeError(msg)
            else:
                msg = "@Task #%i(%s): pass an api to transform or inverse_transform the input data, otherwise you need to fit_transform the data with proper parameters." % (self.iblock, self.SuperFunction)
                raise NameError(msg)

        # step5: process
        available_methods = ['fit_transform', 'transform']
        model, df = self.fit_transform_inverse(model, df, method, available_methods, header=True)

        # step6: send out
        order = [edge[1] for edge in self.Base.graph if edge[0]==self.iblock]
        for token in set(order):
            if token == 'api':
                self.Base.send[(self.iblock, token)] = [model, order.count(token),(self.iblock,token,self.Host,self.Function)]
            elif token == 'df':
                self.Base.send[(self.iblock, token)] = [df, order.count(token),(self.iblock,token,self.Host,self.Function)]
            else:
                msg = "@Task #%i(%s): non valid output token '%s'" % (self.iblock+1, self.SuperFunction, token)
                raise NameError(msg)

        # step7: delete all inputs from memory
        del self.legal_inputs

class Binarizer(BASE,LIBRARY,Preprocessor):
    def legal_IO(self):
        self.legal_inputs = {'df': None, 'api':None}
        self.legal_outputs = {'api':None, 'df':None}
        requirements = ['scikit_learn']
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def fit(self):
        # step1: check inputs
        df, df_info = self.input_check('df', req=True, py_type=pd.DataFrame)
        model, _ = self.input_check('api', req=False)

        # step2: assign inputs to parameters if necessary (param = @token)
        self.paramFROMinput()

        # step3: check the dimension of input data frame
        # df, _ = self.data_check('df', df, ndim=2, n0=None, n1=None, format_out='df')

        # step4: import module and make APIs
        method = self.parameters.pop('func_method')  # method = fit_transform, transform or inverse_transform
        if isinstance(self.legal_inputs['api'], type(None)):
            if method == 'fit_transform':
                try:
                    from sklearn.preprocessing import Binarizer
                    model = Binarizer(**self.parameters)
                except Exception as err:
                    msg = '@Task #%i(%s): '%(self.iblock+1, self.SuperFunction) + type(err).__name__ + ': '+ err.message
                    raise TypeError(msg)
            else:
                msg = "@Task #%i(%s): pass an api to transform or inverse_transform the input data, otherwise you need to fit_transform the data with proper parameters." % (self.iblock, self.SuperFunction)
                raise NameError(msg)

        # step5: process
        available_methods = ['fit_transform', 'transform']
        model, df = self.fit_transform_inverse(model, df, method, available_methods, header=True)

        # step6: send out
        order = [edge[1] for edge in self.Base.graph if edge[0]==self.iblock]
        for token in set(order):
            if token == 'api':
                self.Base.send[(self.iblock, token)] = [model, order.count(token),(self.iblock,token,self.Host,self.Function)]
            elif token == 'df':
                self.Base.send[(self.iblock, token)] = [df, order.count(token),(self.iblock,token,self.Host,self.Function)]
            else:
                msg = "@Task #%i(%s): non valid output token '%s'" % (self.iblock+1, self.SuperFunction, token)
                raise NameError(msg)

        # step7: delete all inputs from memory
        del self.legal_inputs

class OneHotEncoder(BASE,LIBRARY,Preprocessor):
    def legal_IO(self):
        self.legal_inputs = {'df': None, 'api':None}
        self.legal_outputs = {'api':None, 'df':None}
        requirements = ['scikit_learn']
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def fit(self):
        # step1: check inputs
        df, df_info = self.input_check('df', req=True, py_type=pd.DataFrame)
        model, _ = self.input_check('api', req=False)

        # step2: assign inputs to parameters if necessary (param = @token)
        self.paramFROMinput()

        # step3: check the dimension of input data frame
        # df, _ = self.data_check('df', df, ndim=2, n0=None, n1=None, format_out='df')

        # step4: import module and make APIs
        method = self.parameters.pop('func_method')  # method = fit_transform, transform or inverse_transform
        if isinstance(self.legal_inputs['api'], type(None)):
            if method == 'fit_transform':
                try:
                    from sklearn.preprocessing import OneHotEncoder
                    model = OneHotEncoder(**self.parameters)
                except Exception as err:
                    msg = '@Task #%i(%s): ' % (self.iblock + 1, self.SuperFunction) + type(
                        err).__name__ + ': ' + err.message
                    raise TypeError(msg)
            else:
                msg = "@Task #%i(%s): pass an api to transform or inverse_transform the input data, otherwise you need to fit_transform the data with proper parameters." % (
                self.iblock, self.SuperFunction)
                raise NameError(msg)

        # step5: process
        available_methods = ['fit_transform', 'transform']
        model, df = self.fit_transform_inverse(model, df, method, available_methods, header=False)

        # step6: send out
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token == 'api':
                self.Base.send[(self.iblock, token)] = [model, order.count(token),
                                                        (self.iblock, token, self.Host, self.Function)]
            elif token == 'df':
                self.Base.send[(self.iblock, token)] = [df, order.count(token),
                                                        (self.iblock, token, self.Host, self.Function)]
            else:
                msg = "@Task #%i(%s): non valid output token '%s'" % (self.iblock + 1, self.SuperFunction, token)
                raise NameError(msg)

        # step7: delete all inputs from memory
        del self.legal_inputs


# FeatureSelection


# FeatureTransformation

class PCA(BASE,LIBRARY,Preprocessor):
    def legal_IO(self):
        self.legal_inputs = {'df': None, 'api':None}
        self.legal_outputs = {'api':None, 'df':None}
        requirements = ['scikit_learn']
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def fit(self):
        # step1: check inputs
        df, df_info = self.input_check('df', req=True, py_type=pd.DataFrame)
        model, _ = self.input_check('api', req=False)

        # step2: assign inputs to parameters if necessary (param = @token)
        self.paramFROMinput()

        # step3: check the dimension of input data frame
        # df, _ = self.data_check('df', df, ndim=2, n0=None, n1=None, format_out='df')

        # step4: import module and make APIs
        method = self.parameters.pop('func_method')  # method = fit_transform, transform or inverse_transform
        if isinstance(self.legal_inputs['api'], type(None)):
            if method == 'fit_transform':
                try:
                    from sklearn.decomposition import PCA
                    model = PCA(**self.parameters)
                except Exception as err:
                    msg = '@Task #%i(%s): '%(self.iblock+1, self.SuperFunction) + type(err).__name__ + ': '+ err.message
                    raise TypeError(msg)
            else:
                msg = "@Task #%i(%s): pass an api to transform the input data, otherwise you need to fit_transform the data with proper parameters." % (self.iblock, self.SuperFunction)
                raise NameError(msg)

        # step5: process
        available_methods = ['fit_transform', 'transform', 'inverse_transform']
        model, df = self.fit_transform_inverse(model, df, method, available_methods, header=False)

        # step6: send out
        order = [edge[1] for edge in self.Base.graph if edge[0]==self.iblock]
        for token in set(order):
            if token == 'api':
                self.Base.send[(self.iblock, token)] = [model, order.count(token),
                                                        (self.iblock,token,self.Host,self.Function)]
            elif token == 'df':
                self.Base.send[(self.iblock, token)] = [df, order.count(token),
                                                        (self.iblock,token,self.Host,self.Function)]
            else:
                msg = "@Task #%i(%s): non valid output token '%s'" % (self.iblock+1, self.SuperFunction, token)
                raise NameError(msg)

        # step7: delete all inputs from memory
        del self.legal_inputs

class KernelPCA(BASE, LIBRARY, Preprocessor):
    def legal_IO(self):
        self.legal_inputs = {'df': None, 'api': None}
        self.legal_outputs = {'api': None, 'df': None}
        requirements = ['scikit_learn']
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def fit(self):
        # step1: check inputs
        df, df_info = self.input_check('df', req=True, py_type=pd.DataFrame)
        model, _ = self.input_check('api', req=False)

        # step2: assign inputs to parameters if necessary (param = @token)
        self.paramFROMinput()

        # step3: check the dimension of input data frame
        # df, _ = self.data_check('df', df, ndim=2, n0=None, n1=None, format_out='df')

        # step4: import module and make APIs
        method = self.parameters.pop('func_method')  # method = fit_transform, transform or inverse_transform
        if isinstance(self.legal_inputs['api'], type(None)):
            if method == 'fit_transform':
                try:
                    from sklearn.decomposition import KernelPCA
                    model = KernelPCA(**self.parameters)
                except Exception as err:
                    msg = '@Task #%i(%s): ' % (self.iblock + 1, self.SuperFunction) + type(
                        err).__name__ + ': ' + err.message
                    raise TypeError(msg)
            else:
                msg = "@Task #%i(%s): pass an api to transform the input data, otherwise you need to fit_transform the data with proper parameters." % (
                self.iblock, self.SuperFunction)
                raise NameError(msg)

        # step5: process
        available_methods = ['fit_transform', 'transform', 'inverse_transform']
        model, df = self.fit_transform_inverse(model, df, method, available_methods, header=False)

        # step6: send out
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token == 'api':
                self.Base.send[(self.iblock, token)] = [model, order.count(token),
                                                        (self.iblock, token, self.Host, self.Function)]
            elif token == 'df':
                self.Base.send[(self.iblock, token)] = [df, order.count(token),
                                                        (self.iblock, token, self.Host, self.Function)]
            else:
                msg = "@Task #%i(%s): non valid output token '%s'" % (self.iblock + 1, self.SuperFunction, token)
                raise NameError(msg)

        # step7: delete all inputs from memory
        del self.legal_inputs

##################################################################### 3 Define Model

# Regression

class regression(BASE, LIBRARY):
    def legal_IO(self):
        self.legal_inputs = {'dfx': None, 'dfy': None, 'api': None}
        self.legal_outputs = {'api': None, 'dfy_pred':None}
        requirements = ['scikit_learn']
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def fit(self):
        method = self.parameters.pop('func_method')  # method = 'fit', 'predict', None

        # step1: check inputs
        model, _ = self.input_check('api', req=False)
        if method is None:
            dfx, dfx_info = self.input_check('dfx', req=False, py_type=pd.DataFrame)
            dfy, dfy_info = self.input_check('dfy', req=False, py_type=pd.DataFrame)
        else:
            dfx, dfx_info = self.input_check('dfx', req=True, py_type=pd.DataFrame)
        if method == 'fit':
            dfy, dfy_info = self.input_check('dfy', req=True, py_type=pd.DataFrame)

        # step2: assign inputs to parameters if necessary (param = @token)
        self.paramFROMinput()

        # step3: check the dimension of input data frame

        # step4: import module and make APIs
        if model is None:
            if method == 'fit' or method is None:
                try:
                    if self.Function == 'OLS':
                        from sklearn.linear_model import LinearRegression
                        model = LinearRegression(**self.parameters)
                    elif self.Function == 'Ridge':
                        from sklearn.linear_model import Ridge
                        model = Ridge(**self.parameters)
                    elif self.Function == 'KernelRidge':
                        from sklearn.kernel_ridge import KernelRidge
                        model = KernelRidge(**self.parameters)
                    elif self.Function == 'Lasso':
                        from sklearn.linear_model import Lasso
                        model = Lasso(**self.parameters)
                    elif self.Function == 'MultiTaskLasso':
                        from sklearn.linear_model import MultiTaskLasso
                        model = MultiTaskLasso(**self.parameters)
                    elif self.Function == 'ElasticNet':
                        from sklearn.kernel_ridge import ElasticNet
                        model = ElasticNet(**self.parameters)
                    elif self.Function == 'MultiTaskElasticNet':
                        from sklearn.linear_model import MultiTaskElasticNet
                        model = MultiTaskElasticNet(**self.parameters)
                    elif self.Function == 'Lars':
                        from sklearn.linear_model import Lars
                        model = Lars(**self.parameters)
                    elif self.Function == 'LassoLars':
                        from sklearn.kernel_ridge import LassoLars
                        model = LassoLars(**self.parameters)
                    elif self.Function == 'BayesianRidge':
                        from sklearn.linear_model import BayesianRidge
                        model = BayesianRidge(**self.parameters)
                    elif self.Function == 'ARDRegression':
                        from sklearn.linear_model import ARDRegression
                        model = ARDRegression(**self.parameters)
                    elif self.Function == 'LogisticRegression':
                        from sklearn.kernel_ridge import LogisticRegression
                        model = LogisticRegression(**self.parameters)
                    elif self.Function == 'SGDRegressor':
                        from sklearn.linear_model import SGDRegressor
                        model = SGDRegressor(**self.parameters)
                    elif self.Function == 'SVR':
                        from sklearn.svm import SVR
                        model = SVR(**self.parameters)
                    elif self.Function == 'NuSVR':
                        from sklearn.svm import NuSVR
                        model = NuSVR(**self.parameters)
                    elif self.Function == 'LinearSVR':
                        from sklearn.svm import LinearSVR
                        model = LinearSVR(**self.parameters)
                    elif self.Function == 'MLPRegressor':
                        from sklearn.neural_network import MLPRegressor
                        model = MLPRegressor(**self.parameters)
                    else:
                        msg = "@Task #%i(%s): function name '%s' in module '%s' is not an available/valid regression method" % (self.iblock, self.SuperFunction,self.Function, 'sklearn')
                        raise NameError(msg)
                except Exception as err:
                    msg = '@Task #%i(%s): '%(self.iblock+1, self.SuperFunction) + type(err).__name__ + ': '+ err.message
                    raise TypeError(msg)
            else:
                msg = "@Task #%i(%s): pass an api to fit the input data, otherwise you need to fit_transform the data with proper parameters." % (
                    self.iblock, self.SuperFunction)
                raise NameError(msg)

        # step5: process
        if method == 'fit':
            model.fit(dfx, dfy)
        elif method == 'predict':
            dfy_pred = model.predict(dfx)
            dfy_pred = pd.DataFrame(dfy_pred)

        # step6: send out
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token == 'api':
                self.Base.send[(self.iblock, token)] = [model, order.count(token),
                                                        (self.iblock,token,self.Host,self.Function)]
            elif token == 'dfy_pred':
                dfy_pred = model.predict(dfx)
                dfy_pred = pd.DataFrame(dfy_pred)
                self.Base.send[(self.iblock, token)] = [dfy_pred, order.count(token),
                                                        (self.iblock, token, self.Host, self.Function)]
            else:
                msg = "@Task #%i(%s): non valid output token '%s'" % (self.iblock + 1, self.SuperFunction, token)
                raise NameError(msg)
        #step7: delete all inputs from memory
        del self.legal_inputs

##################################################################### 4 Define Search

# Divider

class Train_Test_Split(BASE, LIBRARY):
    def legal_IO(self):
        self.legal_inputs = {'dfx': None, 'dfy': None}
        self.legal_outputs = {'dfx_train': None, 'dfx_test': None, 'dfy_train': None, 'dfy_test': None}
        requirements = ['scikit_learn']
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def fit(self):
        # step1: check inputs
        dfx, dfx_info = self.input_check('dfx', req=True, py_type=pd.DataFrame)
        dfy, dfy_info = self.input_check('dfy', req=False, py_type=pd.DataFrame)

        # step2: assign inputs to parameters if necessary (param = @token)
        self.paramFROMinput()

        # step3: check the dimension of input data frame
        dfx, _ = self.data_check('dfx', dfx, ndim=2, n0=None, n1=None, format_out='df')

        # step4: import module and make APIs
        try:
            from sklearn.model_selection import train_test_split
            if dfy is None:
                tts_out = train_test_split(dfx, **self.parameters)
            else:
                dfy, _ = self.data_check('dfy', dfy, ndim=1, n0=dfx.shape[0], n1=None, format_out='df')
                tts_out = train_test_split(dfx, dfy, **self.parameters)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.SuperFunction) + type(
                err).__name__ + ': ' + err.message
            raise TypeError(msg)

        # step5: process
        # step6: send out
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token == 'dfx_train':
                self.Base.send[(self.iblock, token)] = [tts_out[0], order.count(token),
                                                        (self.iblock,token,self.Host,self.Function)]
            elif token == 'dfx_test':
                self.Base.send[(self.iblock, token)] = [tts_out[1], order.count(token),
                                                        (self.iblock,token,self.Host,self.Function)]
            elif token == 'dfy_train':
                if dfy is None:
                    val = None
                    msg = "@Task #%i(%s): The output for '%s' is None" % (self.iblock + 1, self.SuperFunction, token)
                    warnings.warn(msg)
                else:
                    val = tts_out[2]
                self.Base.send[(self.iblock, token)] = [val, order.count(token),
                                                        (self.iblock,token,self.Host,self.Function)]
            elif token == 'dfy_test':
                if dfy is None:
                    val = None
                    msg = "@Task #%i(%s): The output for '%s' is None" % (self.iblock + 1, self.SuperFunction, token)
                    warnings.warn(msg)
                else:
                    val = tts_out[3]
                self.Base.send[(self.iblock, token)] = [val, order.count(token),
                                                        (self.iblock,token,self.Host,self.Function)]
            else:
                msg = "@Task #%i(%s): non valid output token '%s'" % (self.iblock + 1, self.SuperFunction, token)
                raise NameError(msg)

        # step7: delete all inputs from memory
        del self.legal_inputs

class KFold(BASE, LIBRARY):
    def legal_IO(self):
        self.legal_inputs = {}
        self.legal_outputs = {'kfold': None}
        requirements = ['scikit_learn']
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def fit(self):
        # step1: check inputs
        # step2: assign inputs to parameters if necessary (param = @token)
        self.paramFROMinput()

        # step3: check the dimension of input data frame
        # step4: import module and make APIs
        try:
            from sklearn.model_selection import KFold
            model = KFold(**self.parameters)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.SuperFunction) + type(
                err).__name__ + ': ' + err.message
            raise TypeError(msg)

        # step5: process
        # step6: send out
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token == 'kfold':
                self.Base.send[(self.iblock, token)] = [model, order.count(token),
                                                        (self.iblock,token,self.Host,self.Function)]
            else:
                msg = "@Task #%i(%s): non valid output token '%s'" % (self.iblock + 1, self.SuperFunction, token)
                raise NameError(msg)
        # step7: delete all inputs from memory
        del self.legal_inputs

class GridSearchCV(BASE, LIBRARY):
    def legal_IO(self):
        self.legal_inputs = {'dfx': None, 'dfy': None, 'estimator': None}
        self.legal_outputs = {'cv_results_': None, 'api': None, 'best_estimator_': None}
        requirements = ['scikit_learn']
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def fit(self):
        # step1: check inputs
        _, _ = self.input_check('estimator', req = True)
        dfx, dfx_info = self.input_check('dfx', req = True, py_type = pd.DataFrame)
        dfy, dfy_info = self.input_check('dfy', req = True, py_type = pd.DataFrame)

        # step2: assign inputs to parameters if necessary (param = @token)
        self.paramFROMinput()

        # step3: check the dimension of input data frame
        dfx, _ = self.data_check('dfx', dfx, ndim=2, n0=None, n1=None, format_out='df')
        dfy, _ = self.data_check('dfy', dfy, ndim=1, n0=dfx.shape[0], n1=None, format_out='df')

        # step4: import module and make APIs
        try:
            from sklearn.model_selection import GridSearchCV
            api = GridSearchCV(**self.parameters)
        except Exception as err:
            msg = '@Task #%i(%s): '%(self.iblock+1, self.SuperFunction) + type(err).__name__ + ': '+ err.message
            raise TypeError(msg)

        # step5: process
        api.fit(dfx, dfy)

        # step6: send out
        order = [edge[1] for edge in self.Base.graph if edge[0]==self.iblock]
        for token in set(order):
            if token == 'best_estimator_':
                if self.parameters['refit']==True:
                    best_estimator_ = copy.deepcopy(api.best_estimator_)
                else:
                    best_estimator_ = copy.deepcopy(self.parameters['estimator'])
                    best_estimator_.set_params(**api.best_params_)
                    # best_estimator_.fit(dfx,dfy)
                self.Base.send[(self.iblock, token)] = [best_estimator_, order.count(token),
                                                        (self.iblock,token,self.Host,self.Function)]
            elif token == 'cv_results_':
                self.Base.send[(self.iblock, token)] = [pd.DataFrame(api.cv_results_), order.count(token),
                                                        (self.iblock,token,self.Host,self.Function)]
            elif token == 'api':
                self.Base.send[(self.iblock, token)] = [api, order.count(token),
                                                        (self.iblock,token,self.Host,self.Function)]
            else:
                msg = "@Task #%i(%s): non valid output token '%s'" % (self.iblock+1, self.SuperFunction, token)
                raise NameError(msg)

        #step7: delete all inputs from memory
        del self.legal_inputs

class cross_val_score(BASE, LIBRARY):
    def legal_IO(self):
        self.legal_inputs = {'dfx': None, 'dfy': None, 'estimator': None}
        self.legal_outputs = {'scores': None}
        requirements = ['scikit_learn']
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def fit(self):
        # step1: check inputs
        _, _ = self.input_check('estimator', req=True)
        dfx, dfx_info = self.input_check('dfx', req=True, py_type=pd.DataFrame)
        dfy, dfy_info = self.input_check('dfy', req=False, py_type=pd.DataFrame)

        # step2: assign inputs to parameters if necessary (param = @token)
        self.paramFROMinput()

        # step3: check the dimension of input data frame
        dfx, _ = self.data_check('dfx', dfx, ndim=2, n0=None, n1=None, format_out='df')

        # step4: import module and make APIs
        try:
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(X=dfx, y=dfy, **self.parameters)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.SuperFunction) + type(
                err).__name__ + ': ' + err.message
            raise TypeError(msg)

        # step5: process
        # step6: send out
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token == 'scores':
                self.Base.send[(self.iblock, token)] = [scores, order.count(token),
                                                        (self.iblock,token,self.Host,self.Function)]
            else:
                msg = "@Task #%i(%s): non valid output token '%s'" % (self.iblock + 1, self.SuperFunction, token)
                raise NameError(msg)

        # step7: delete all inputs from memory
        del self.legal_inputs


# Metrics

class Evaluate_Regression(BASE,LIBRARY,Evaluator):
    def legal_IO(self):
        self.legal_inputs = {'dfy': None, 'dfy_pred': None}
        self.legal_outputs = {'evaluation_results_': None, 'evaluator': None}
        requirements = ['scikit_learn']
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def fit(self):
        # step2: assign inputs to parameters if necessary (param = @token)
        self.paramFROMinput()

        # step4: import module and make APIs
        try:
            self._reg_evaluation_params()
        except Exception as err:
            msg = '@Task #%i(%s): '%(self.iblock+1, self.SuperFunction) + type(err).__name__ + ': '+ err.message
            raise TypeError(msg)

        # step5: process
        # step6: send out
        order = [edge[1] for edge in self.Base.graph if edge[0]==self.iblock]
        for token in set(order):
            if token == 'evaluator':
                self.Base.send[(self.iblock, token)] = [self.evaluator, order.count(token),
                                                        (self.iblock, token, self.Host, self.Function)]
            elif token == 'evaluation_results_':
                # step1: check inputs
                dfy, dfy_info = self.input_check('dfy', req=True, py_type=pd.DataFrame)
                dfy_pred, dfy_pred_info = self.input_check('dfy_pred', req=True, py_type=pd.DataFrame)

                # step3: check the dimension of input data frame
                dfy, _ = self.data_check('dfy', dfy, ndim=2, n0=None, n1=None, format_out='df')
                dfy_pred, _ = self.data_check('dfy_pred', dfy_pred, ndim=2, n0=dfy.shape[0], n1=None, format_out='df')

                self._reg_evaluate(dfy, dfy_pred, self.evaluator)
                evaluation_results_ = self.results
                self.Base.send[(self.iblock, token)] = [pd.DataFrame(evaluation_results_), order.count(token),
                                                        (self.iblock,token,self.Host,self.Function)]
            else:
                msg = "@Task #%i(%s): non valid output token '%s'" % (self.iblock+1, self.SuperFunction, token)
                raise NameError(msg)

        #step7: delete all inputs from memory
        del self.legal_inputs

##################################################################### 5 Train/Run

class Search(BASE, Regressor, Evaluator):
    def legal_IO(self):
        self.legal_inputs = {'dfx': None, 'dfy': None, 'CV': None,'model': None, }
        self.legal_outputs = {'results': None}
        requirements = ['scikit_learn']
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def scenario1(self, dfx_test, dfy_test):
        # 0 CV
        if self.X_scaler is not None:
            dfx_test = self.X_scaler.transform(dfx_test)
        if len(dfy_test.columns)==1:
            col = dfy_test.columns[0]
            dfy_test = np.array(dfy_test[col])
        self.dfy_pred = self.predict(self.legal_inputs['model'], dfx_test)
        if self.Y_scaler is not None:
            self.dfy_pred = self.Y_scaler.inverse_transform(self.dfy_pred)
        self.evaluate(dfy_test, self.dfy_pred)

    def scenario2(self, dfx, dfy):
        # 1 CV
        self.model_info = {'r2_training':[], 'models':[], 'X_scaler':[], 'Y_scaler':[]}
        for train, test in self.CV.split(dfx):
            dfx_train = dfx.iloc[train]
            dfy_train = dfy.iloc[train]
            if dfy_train.shape[1]==1:
                flag = True
            else:
                flag = False
            dfx_test = dfx.iloc[test]
            dfy_test = dfy.iloc[test]
            if not isinstance(self.X_scaler, type(None)):
                dfx_train = self.X_scaler.fit_transform(dfx_train)
                dfx_test = self.X_scaler.transform(dfx_test)
                self.model_info['X_scaler'].append(self.X_scaler)
            if not isinstance(self.Y_scaler, type(None)):
                dfy_train = self.Y_scaler.fit_transform(dfy_train)
                self.model_info['Y_scaler'].append(self.Y_scaler)
                print dfy_train.shape
            if flag:
                dfy_train = np.ravel(dfy_train)
            model, r2 = self.train(self.legal_inputs['model'], dfx_train, dfy_train)
            self.model_info['models'].append(model)
            self.model_info['r2_training'].append(r2)
            dfy_pred = self.predict(self.legal_inputs['model'], dfx_test)
            if not isinstance(self.Y_scaler, type(None)):
                dfy_pred = self.Y_scaler.inverse_transform(dfy_pred)
            self.evaluate(dfy_test, dfy_pred)

    def fit(self):
        #Todo: add a function to read the model from file
        dfx = self.type_check('dfx', cheml_type='df', req=True, py_type=pd.DataFrame)
        dfy = self.type_check('dfy', cheml_type='df', req=True, py_type=pd.DataFrame)
        model = self.type_check('model', cheml_type='model', req=True)
        self.X_scaler = self.type_check('X_scaler', cheml_type='preprocessor', req=False)
        self.Y_scaler = self.type_check('Y_scaler', cheml_type='preprocessor', req=False)
        self.CV = self.type_check('CV', cheml_type='cv', req=False)

        self._evaluation_params()
        # Todo: check if we can have one sent variable for two inputs
        if self.CV is None:
            self.scenario1(dfx, dfy)
        else:
            self.scenario2(dfx, dfy)
        del dfx
        del dfy

        order = [edge[1] for edge in self.Base.graph if edge[0]==self.iblock]
        for token in set(order):
            if token == 'results':
                self.Base.send[(self.iblock, token)] = [pd.DataFrame(self.results), order.count(token)]
            else:
                msg = "@Task #%i(%s): non valid output token '%s'" % (self.iblock+1, self.SuperFunction, token)
                raise NameError(msg)
        del self.legal_inputs
