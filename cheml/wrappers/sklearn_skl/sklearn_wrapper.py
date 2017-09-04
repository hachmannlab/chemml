import warnings
import pandas as pd
import numpy as np
import sklearn
import copy

from ..base import BASE, LIBRARY
from .syntax import Preprocessor, Regressor, Evaluator


##################################################################### 2 Prepare Data

# descriptor

class PolynomialFeatures(BASE,LIBRARY,Preprocessor):
    def legal_IO(self):
        self.legal_inputs = {'df': None, 'api':None}
        self.legal_outputs = {'df':None, 'api':None}
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

class Binarizer(BASE, LIBRARY, Preprocessor):
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
                    from sklearn.preprocessing import Binarizer
                    model = Binarizer(**self.parameters)
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
        model, df = self.fit_transform_inverse(model, df, method, available_methods, header=True)

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

class OneHotEncoder(BASE, LIBRARY, Preprocessor):
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


# basic oprator

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

# scaler

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

# feature selector

# feature transformer

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

# splitter

class train_test_split(BASE, LIBRARY):
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
        self.legal_outputs = {'cv': None}
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
            if token == 'cv':
                self.Base.send[(self.iblock, token)] = [model, order.count(token),
                                                        (self.iblock,token,self.Host,self.Function)]
            else:
                msg = "@Task #%i(%s): non valid output token '%s'" % (self.iblock + 1, self.SuperFunction, token)
                raise NameError(msg)
        # step7: delete all inputs from memory
        del self.legal_inputs

class ShuffleSplit(BASE, LIBRARY):
    def legal_IO(self):
        self.legal_inputs = {}
        self.legal_outputs = {'cv': None}
        requirements = ['scikit_learn']
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def fit(self):
        # step1: check inputs
        # step2: assign inputs to parameters if necessary (param = @token)
        self.paramFROMinput()

        # step3: check the dimension of input data frame
        # step4: import module and make APIs
        try:
            from sklearn.model_selection import ShuffleSplit
            model = ShuffleSplit(**self.parameters)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.SuperFunction) + type(
                err).__name__ + ': ' + err.message
            raise TypeError(msg)

        # step5: process
        # step6: send out
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token == 'cv':
                self.Base.send[(self.iblock, token)] = [model, order.count(token),
                                                        (self.iblock, token, self.Host, self.Function)]
            else:
                msg = "@Task #%i(%s): non valid output token '%s'" % (self.iblock + 1, self.SuperFunction, token)
                raise NameError(msg)
        # step7: delete all inputs from memory
        del self.legal_inputs

class StratifiedShuffleSplit(BASE, LIBRARY):
    def legal_IO(self):
        self.legal_inputs = {}
        self.legal_outputs = {'cv': None}
        requirements = ['scikit_learn']
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def fit(self):
        # step1: check inputs
        # step2: assign inputs to parameters if necessary (param = @token)
        self.paramFROMinput()

        # step3: check the dimension of input data frame
        # step4: import module and make APIs
        try:
            from sklearn.model_selection import StratifiedShuffleSplit
            model = StratifiedShuffleSplit(**self.parameters)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.SuperFunction) + type(
                err).__name__ + ': ' + err.message
            raise TypeError(msg)

        # step5: process
        # step6: send out
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token == 'cv':
                self.Base.send[(self.iblock, token)] = [model, order.count(token),
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
            model, _ = self.input_check('api', req=False)
            dfy, dfy_info = self.input_check('dfy', req=True, py_type=pd.DataFrame)
            # step3: check the dimension of input data frame
            dfy, _ = self.data_check('dfy', dfy, ndim=1, n0=dfx.shape[0], n1=None, format_out='ar')
        if method == 'predict':
            model, _ = self.input_check('api', req=True)

        # step2: assign inputs to parameters if necessary (param = @token)
        self.paramFROMinput()

        # step3: check the dimension of input data frame
        # dfy, _ = self.data_check('dfy', dfy, ndim=1, n0=dfx.shape[0], n1=None, format_out='ar')

        # step4: import module and make APIs
        if model is None:
            if method == 'fit' or method is None:
                try:
                    if self.Function == 'LinearRegression':
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
                        from sklearn.linear_model import ElasticNet
                        model = ElasticNet(**self.parameters)
                    elif self.Function == 'MultiTaskElasticNet':
                        from sklearn.linear_model import MultiTaskElasticNet
                        model = MultiTaskElasticNet(**self.parameters)
                    elif self.Function == 'Lars':
                        from sklearn.linear_model import Lars
                        model = Lars(**self.parameters)
                    elif self.Function == 'LassoLars':
                        from sklearn.linear_model import LassoLars
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
                if method not in ['fit', 'predict']:
                    msg = "@Task #%i(%s): no calculation for dfy_pred has been requested, func_method = %s" % (self.iblock + 1, self.SuperFunction, str(method))
                    raise ValueError(msg)
                if method == 'fit':
                    dfy_pred = model.predict(dfx)
                    dfy_pred = pd.DataFrame(dfy_pred)
                self.Base.send[(self.iblock, token)] = [dfy_pred, order.count(token),
                                                        (self.iblock, token, self.Host, self.Function)]
            else:
                msg = "@Task #%i(%s): non valid output token '%s'" % (self.iblock + 1, self.SuperFunction, token)
                raise NameError(msg)
        #step7: delete all inputs from memory
        del self.legal_inputs

##################################################################### 4 Search

class GridSearchCV(BASE, LIBRARY):
    def legal_IO(self):
        self.legal_inputs = {'dfx': None, 'dfy': None, 'estimator': None, 'scorer':None}
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
        dfy, _ = self.data_check('dfy', dfy, ndim=1, n0=dfx.shape[0], n1=None, format_out='ar')

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

class cross_val_predict(BASE, LIBRARY):
    def legal_IO(self):
        self.legal_inputs = {'dfx': None, 'dfy': None, 'estimator': None}
        self.legal_outputs = {'dfy_pred': None}
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
            from sklearn.model_selection import cross_val_predict
            predictions = cross_val_score(X=dfx, y=dfy, **self.parameters)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.SuperFunction) + type(
                err).__name__ + ': ' + err.message
            raise TypeError(msg)

        # step5: process
        # step6: send out
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token == 'dfy_pred':
                self.Base.send[(self.iblock, token)] = [predictions, order.count(token),
                                                        (self.iblock, token, self.Host, self.Function)]
            else:
                msg = "@Task #%i(%s): non valid output token '%s'" % (self.iblock + 1, self.SuperFunction, token)
                raise NameError(msg)

        # step7: delete all inputs from memory
        del self.legal_inputs

class learning_curve(BASE, LIBRARY):
    def legal_IO(self):
        self.legal_inputs = {'dfx': None, 'dfy': None, 'estimator': None, 'cv':None, 'scorer':None}
        self.legal_outputs = {'train_sizes_abs': None,'train_scores': None,'test_scores': None,
                              'extended_result_': None}
        requirements = ['scikit_learn']
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def fit(self):
        # step1: check inputs
        _, _ = self.input_check('estimator', req=True)
        dfx, dfx_info = self.input_check('dfx', req=True, py_type=pd.DataFrame)
        dfy, dfy_info = self.input_check('dfy', req=False, py_type=pd.DataFrame)

        # step2: assign inputs to parameters if necessary (param = @token)
        self.paramFROMinput()
        estimator = self.parameters.pop('estimator')
        if type(self.parameters['train_sizes']) is str:
            self.parameters['train_sizes'] = eval(self.parameters['train_sizes'])

        # step3: check the dimension of input data frame
        dfx, _ = self.data_check('dfx', dfx, ndim=2, n0=None, n1=None, format_out='df')
        if dfy is not None:
            dfy, _ = self.data_check('dfy', dfy, ndim=1, n0=dfx.shape[0], n1=None, format_out='ar')

        # step4: import module and make APIs
        try:
            from sklearn.model_selection import learning_curve
            train_sizes_abs, train_scores, test_scores = learning_curve(estimator=estimator,X=dfx,y=dfy,**self.parameters)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.SuperFunction) + type(
                err).__name__ + ': ' + err.message
            raise TypeError(msg)

        # step5: process
        # step6: send out
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token == 'train_sizes_abs':
                self.Base.send[(self.iblock, token)] = [train_sizes_abs, order.count(token),
                                                        (self.iblock, token, self.Host, self.Function)]
            elif token == 'train_scores':
                self.Base.send[(self.iblock, token)] = [train_scores, order.count(token),
                                                        (self.iblock, token, self.Host, self.Function)]
            elif token == 'test_scores':
                self.Base.send[(self.iblock, token)] = [test_scores, order.count(token),
                                                        (self.iblock, token, self.Host, self.Function)]
            elif token == 'extended_result_':
                train_scores_mean = np.mean(train_scores, axis=1)
                train_scores_std = np.std(train_scores, axis=1)
                test_scores_mean = np.mean(test_scores, axis=1)
                test_scores_std = np.std(test_scores, axis=1)
                data = [train_sizes_abs, train_scores_mean, train_scores_std, test_scores_mean, test_scores_std]
                cols = ['train_sizes', 'train_scores_mean', 'train_scores_std', 'test_scores_mean', 'test_scores_std']
                extended_result_ = pd.DataFrame()
                for i, col in enumerate(cols):
                    extended_result_[col] = data[i]

                self.Base.send[(self.iblock, token)] = [extended_result_, order.count(token),
                                                        (self.iblock, token, self.Host, self.Function)]
            else:
                msg = "@Task #%i(%s): non valid output token '%s'" % (self.iblock + 1, self.SuperFunction, token)
                raise NameError(msg)

        # step7: delete all inputs from memory
        del self.legal_inputs


# Metrics

class evaluate_regression(BASE,LIBRARY,Evaluator):
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

class scorer_regression(BASE, LIBRARY):
    def legal_IO(self):
        self.legal_inputs = {}
        self.legal_outputs = {'scorer': None}
        requirements = ['scikit_learn']
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def fit(self):
        # step1: check inputs
        # step2: assign inputs to parameters if necessary (param = @token)
        self.paramFROMinput()
        if 'metric' in self.parameters:
            metric = self.parameters.pop('metric')
        else:
            metric = 'mae'
        # Todo: add all the metrics for regression

        # step3: check the dimension of input data frame
        # step4: import module and make APIs
        try:
            from sklearn.metrics import make_scorer
            if metric == 'mae':
                from sklearn.metrics import mean_absolute_error
                scorer = make_scorer(mean_absolute_error,**self.parameters)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.SuperFunction) + type(
                err).__name__ + ': ' + err.message
            raise TypeError(msg)

        # step5: process
        # step6: send out
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token == 'scorer':
                self.Base.send[(self.iblock, token)] = [scorer, order.count(token),
                                                        (self.iblock, token, self.Host, self.Function)]
            else:
                msg = "@Task #%i(%s): non valid output token '%s'" % (self.iblock + 1, self.SuperFunction, token)
                raise NameError(msg)
        # step7: delete all inputs from memory
        del self.legal_inputs


##################################################################### 5 Mix

