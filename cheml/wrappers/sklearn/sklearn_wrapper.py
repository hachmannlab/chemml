import warnings
import pandas as pd
import numpy as np
import sklearn
import copy

from ..base import BASE, LIBRARY
from .syntax import Preprocessor, Regressor, Evaluator


#####################################################################DataRepresentation

class PolynomialFeatures(BASE,Preprocessor):
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

#####################################################################Preprocessor

class Imputer(BASE,Preprocessor):
    def legal_IO(self):
        self.legal_inputs = {'df': None}
        self.legal_outputs = {'api':None, 'df':None}
        requirements = ['scikit_learn']
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def fit(self):
        from sklearn.preprocessing import Imputer
        cheml_type = "%s_%s" % (self.Base.graph_info[self.iblock][0], self.Base.graph_info[self.iblock][1])
        self.Base.cheml_type['preprocessor'].append(cheml_type)
        try:
            model = Imputer(**self.parameters)
        except Exception as err:
            msg = '@Task #%i(%s): '%(self.iblock+1, self.SuperFunction) + type(err).__name__ + ': '+ err.message
            raise TypeError(msg)
        # Send

        if not isinstance(self.legal_inputs['df'],type(None)):
            df = self.legal_inputs['df'][0]
            model, df = self.Imputer_ManipulateHeader(model, df)

        order = [edge[1] for edge in self.Base.graph if edge[0]==self.iblock]
        for token in set(order):
            if token == 'api':
                val = model
                self.Base.send[(self.iblock, token)] = [val, order.count(token)]
            elif token == 'df':
                if not isinstance(self.legal_inputs['df'], type(None)):
                    self.Base.send[(self.iblock, token)] = [df, order.count(token)]
            else:
                msg = "@Task #%i(%s): non valid output token '%s'" % (self.iblock+1, self.SuperFunction, token)
                raise NameError(msg)
        del self.legal_inputs

class StandardScaler(BASE,Preprocessor):
    def legal_IO(self):
        self.legal_inputs = {'df': None}
        self.legal_outputs = {'api':None, 'df':None}
        requirements = ['scikit_learn']
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def fit(self):
        from sklearn.preprocessing import StandardScaler
        cheml_type = "%s_%s" % (self.Base.graph_info[self.iblock][0], self.Base.graph_info[self.iblock][1])
        self.Base.cheml_type['preprocessor'].append(cheml_type)
        try:
            model = StandardScaler(**self.parameters)
        except Exception as err:
            msg = '@Task #%i(%s): '%(self.iblock+1, self.SuperFunction) + type(err).__name__ + ': '+ err.message
            raise TypeError(msg)

        if not isinstance(self.legal_inputs['df'],type(None)):
            df = self.legal_inputs['df'][0]
            model, df = self.Transformer_ManipulateHeader(model, df)

        order = [edge[1] for edge in self.Base.graph if edge[0]==self.iblock]
        for token in set(order):
            if token == 'api':
                val = model
                self.Base.send[(self.iblock, token)] = [val, order.count(token)]
            elif token == 'df':
                if not isinstance(self.legal_inputs['df'], type(None)):
                    self.Base.send[(self.iblock, token)] = [df, order.count(token)]
            else:
                msg = "@Task #%i(%s): non valid output token '%s'" % (self.iblock, self.SuperFunction, token)
                raise NameError(msg)
        del self.legal_inputs

class MinMaxScaler(BASE,Preprocessor):
    def legal_IO(self):
        self.legal_inputs = {'df': None}
        self.legal_outputs = {'api':None, 'df':None}
        requirements = ['scikit_learn']
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def fit(self):
        from sklearn.preprocessing import MinMaxScaler
        cheml_type = "%s_%s" % (self.Base.graph_info[self.iblock][0], self.Base.graph_info[self.iblock][1])
        self.Base.cheml_type['preprocessor'].append(cheml_type)
        try:
            model = MinMaxScaler(**self.parameters)
        except Exception as err:
            msg = '@Task #%i(%s): '%(self.iblock+1, self.SuperFunction) + type(err).__name__ + ': '+ err.message
            raise TypeError(msg)

        if not isinstance(self.legal_inputs['df'],type(None)):
            df = self.legal_inputs['df'][0]
            model, df = self.Transformer_ManipulateHeader(model, df)

        order = [edge[1] for edge in self.Base.graph if edge[0]==self.iblock]
        for token in set(order):
            if token == 'api':
                val = model
                self.Base.send[(self.iblock, token)] = [val, order.count(token)]
            elif token == 'df':
                if not isinstance(self.legal_inputs['df'], type(None)):
                    self.Base.send[(self.iblock, token)] = [df, order.count(token)]
            else:
                msg = "@Task #%i(%s): non valid output token '%s'" % (self.iblock+1, self.SuperFunction, token)
                raise NameError(msg)
        del self.legal_inputs

class MaxAbsScaler(BASE,Preprocessor):
    def legal_IO(self):
        self.legal_inputs = {'df': None}
        self.legal_outputs = {'api':None, 'df':None}
        requirements = ['scikit_learn']
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def fit(self):
        from sklearn.preprocessing import MaxAbsScaler
        cheml_type = "%s_%s" % (self.Base.graph_info[self.iblock][0], self.Base.graph_info[self.iblock][1])
        self.Base.cheml_type['preprocessor'].append(cheml_type)
        try:
            model = MaxAbsScaler(**self.parameters)
        except Exception as err:
            msg = '@Task #%i(%s): '%(self.iblock+1, self.SuperFunction) + type(err).__name__ + ': '+ err.message
            raise TypeError(msg)

        if not isinstance(self.legal_inputs['df'],type(None)):
            df = self.legal_inputs['df'][0]
            model, df = self.Transformer_ManipulateHeader(model, df)

        order = [edge[1] for edge in self.Base.graph if edge[0]==self.iblock]
        for token in set(order):
            if token == 'api':
                val = model
                self.Base.send[(self.iblock, token)] = [val, order.count(token)]
            elif token == 'df':
                if not isinstance(self.legal_inputs['df'], type(None)):
                    self.Base.send[(self.iblock, token)] = [df, order.count(token)]
            else:
                msg = "@Task #%i(%s): non valid output token '%s'" % (self.iblock+1, self.SuperFunction, token)
                raise NameError(msg)
        del self.legal_inputs

class RobustScaler(BASE,Preprocessor):
    def legal_IO(self):
        self.legal_inputs = {'df': None}
        self.legal_outputs = {'api':None, 'df':None}
        requirements = ['scikit_learn']
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def fit(self):
        from sklearn.preprocessing import RobustScaler
        cheml_type = "%s_%s" % (self.Base.graph_info[self.iblock][0], self.Base.graph_info[self.iblock][1])
        self.Base.cheml_type['preprocessor'].append(cheml_type)
        try:
            model = RobustScaler(**self.parameters)
        except Exception as err:
            msg = '@Task #%i(%s): '%(self.iblock+1, self.SuperFunction) + type(err).__name__ + ': '+ err.message
            raise TypeError(msg)

        if not isinstance(self.legal_inputs['df'],type(None)):
            df = self.legal_inputs['df'][0]
            model, df = self.Transformer_ManipulateHeader(model, df)

        order = [edge[1] for edge in self.Base.graph if edge[0]==self.iblock]
        for token in set(order):
            if token == 'api':
                val = model
                self.Base.send[(self.iblock, token)] = [val, order.count(token)]
            elif token == 'df':
                if not isinstance(self.legal_inputs['df'], type(None)):
                    self.Base.send[(self.iblock, token)] = [df, order.count(token)]
            else:
                msg = "@Task #%i(%s): non valid output token '%s'" % (self.iblock+1, self.SuperFunction, token)
                raise NameError(msg)
        del self.legal_inputs

class Normalizer(BASE,Preprocessor):
    def legal_IO(self):
        self.legal_inputs = {'df': None}
        self.legal_outputs = {'api':None, 'df':None}
        requirements = ['scikit_learn']
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def fit(self):
        from sklearn.preprocessing import Normalizer
        cheml_type = "%s_%s" % (self.Base.graph_info[self.iblock][0], self.Base.graph_info[self.iblock][1])
        self.Base.cheml_type['preprocessor'].append(cheml_type)
        try:
            model = Normalizer(**self.parameters)
        except Exception as err:
            msg = '@Task #%i(%s): '%(self.iblock+1, self.SuperFunction) + type(err).__name__ + ': '+ err.message
            raise TypeError(msg)

        if not isinstance(self.legal_inputs['df'],type(None)):
            df = self.legal_inputs['df'][0]
            model, df = self.Transformer_ManipulateHeader(model, df)

        order = [edge[1] for edge in self.Base.graph if edge[0]==self.iblock]
        for token in set(order):
            if token == 'api':
                val = model
                self.Base.send[(self.iblock, token)] = [val, order.count(token)]
            elif token == 'df':
                if not isinstance(self.legal_inputs['df'], type(None)):
                    self.Base.send[(self.iblock, token)] = [df, order.count(token)]
            else:
                msg = "@Task #%i(%s): non valid output token '%s'" % (self.iblock+1, self.SuperFunction, token)
                raise NameError(msg)
        del self.legal_inputs

class Binarizer(BASE,Preprocessor):
    def legal_IO(self):
        self.legal_inputs = {'df': None}
        self.legal_outputs = {'api':None, 'df':None}
        requirements = ['scikit_learn']
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def fit(self):
        from sklearn.preprocessing import Binarizer
        cheml_type = "%s_%s" % (self.Base.graph_info[self.iblock][0], self.Base.graph_info[self.iblock][1])
        self.Base.cheml_type['preprocessor'].append(cheml_type)
        try:
            model = Binarizer(**self.parameters)
        except Exception as err:
            msg = '@Task #%i(%s): '%(self.iblock+1, self.SuperFunction) + type(err).__name__ + ': '+ err.message
            raise TypeError(msg)

        if not isinstance(self.legal_inputs['df'],type(None)):
            df = self.legal_inputs['df'][0]
            model, df = self.Transformer_ManipulateHeader(model, df)

        order = [edge[1] for edge in self.Base.graph if edge[0]==self.iblock]
        for token in set(order):
            if token == 'api':
                val = model
                self.Base.send[(self.iblock, token)] = [val, order.count(token)]
            elif token == 'df':
                if not isinstance(self.legal_inputs['df'], type(None)):
                    self.Base.send[(self.iblock, token)] = [df, order.count(token)]
            else:
                msg = "@Task #%i(%s): non valid output token '%s'" % (self.iblock+1, self.SuperFunction, token)
                raise NameError(msg)
        del self.legal_inputs

class OneHotEncoder(BASE,Preprocessor):
    def legal_IO(self):
        self.legal_inputs = {'df': None}
        self.legal_outputs = {'api':None, 'df':None}
        requirements = ['scikit_learn']
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def fit(self):
        from sklearn.preprocessing import OneHotEncoder
        cheml_type = "%s_%s" % (self.Base.graph_info[self.iblock][0], self.Base.graph_info[self.iblock][1])
        self.Base.cheml_type['preprocessor'].append(cheml_type)
        try:
            model = OneHotEncoder(**self.parameters)
        except Exception as err:
            msg = '@Task #%i(%s): '%(self.iblock+1, self.SuperFunction) + type(err).__name__ + ': '+ err.message
            raise TypeError(msg)

        if not isinstance(self.legal_inputs['df'],type(None)):
            df = self.legal_inputs['df'][0]
            model, df = self.Transformer_ManipulateHeader(model, df)

        order = [edge[1] for edge in self.Base.graph if edge[0]==self.iblock]
        for token in set(order):
            if token == 'api':
                val = model
                self.Base.send[(self.iblock, token)] = [val, order.count(token)]
            elif token == 'df':
                if not isinstance(self.legal_inputs['df'], type(None)):
                    self.Base.send[(self.iblock, token)] = [df, order.count(token)]
            else:
                msg = "@Task #%i(%s): non valid output token '%s'" % (self.iblock+1, self.SuperFunction, token)
                raise NameError(msg)
        del self.legal_inputs

#####################################################################FeatureSelection



#####################################################################FeatureTransformation

class PCA(BASE):
    def legal_IO(self):
        self.legal_inputs = {'df': None}
        self.legal_outputs = {'api':None, 'df':None}
        requirements = ['scikit_learn']
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def fit(self):
        from sklearn.decomposition import PCA
        cheml_type = "%s_%s" % (self.Base.graph_info[self.iblock][0], self.Base.graph_info[self.iblock][1])
        self.Base.cheml_type['transformer'].append(cheml_type)
        try:
            model = PCA(**self.parameters)
        except Exception as err:
            msg = '@Task #%i(%s): '%(self.iblock+1, self.SuperFunction) + type(err).__name__ + ': '+ err.message
            raise TypeError(msg)
        order = [edge[1] for edge in self.Base.graph if edge[0]==self.iblock]
        for token in set(order):
            if token == 'api':
                val = model
                self.Base.send[(self.iblock, token)] = [val, order.count(token)]
            elif token == 'df':
                val = pd.DataFrame(model.fit_transform(self.legal_inputs['df'][0]))
                self.Base.send[(self.iblock, token)] = [val, order.count(token)]
            else:
                msg = "@Task #%i(%s): non valid output token '%s'" % (self.iblock+1, self.SuperFunction, token)
                raise NameError(msg)
        del self.legal_inputs

#####################################################################Divider

class Train_Test_Split(BASE):
    def legal_IO(self):
        self.legal_inputs = {'dfx': None, 'dfy': None}
        self.legal_outputs = {'dfx_train': None, 'dfx_test': None, 'dfy_train': None, 'dfy_test': None}
        requirements = ['scikit_learn']
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def fit(self):
        dfx = self.type_check('dfx', cheml_type='df', req=True, py_type=pd.DataFrame)
        dfy = self.type_check('dfy', cheml_type='df', req=False, py_type=pd.DataFrame)

        from sklearn.model_selection import train_test_split
        cheml_type = "%s_%s" % (self.Base.graph_info[self.iblock][0], self.Base.graph_info[self.iblock][1])
        self.Base.cheml_type['transformer'].append(cheml_type)
        try:
            if dfy is None:
                tts_out = train_test_split(dfx,**self.parameters)
            else:
                tts_out = train_test_split(dfx,dfy, **self.parameters)
        except Exception as err:
            msg = '@Task #%i(%s): '%(self.iblock+1, self.SuperFunction) + type(err).__name__ + ': '+ err.message
            raise TypeError(msg)
        order = [edge[1] for edge in self.Base.graph if edge[0]==self.iblock]
        for token in set(order):
            if token == 'dfx_train':
                self.Base.send[(self.iblock, token)] = [tts_out[0], order.count(token)]
            elif token == 'dfx_test':
                self.Base.send[(self.iblock, token)] = [tts_out[1], order.count(token)]
            elif token == 'dfy_train':
                self.Base.send[(self.iblock, token)] = [tts_out[2], order.count(token)]
            elif token == 'dfy_test':
                self.Base.send[(self.iblock, token)] = [tts_out[3], order.count(token)]
            else:
                msg = "@Task #%i(%s): non valid output token '%s'" % (self.iblock+1, self.SuperFunction, token)
                raise NameError(msg)
        del self.legal_inputs

class KFold(BASE):
    def legal_IO(self):
        self.legal_inputs = {}
        self.legal_outputs = {'CV': None}
        requirements = ['scikit_learn']
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def fit(self):
        from sklearn.model_selection import KFold
        cheml_type = "%s_%s" % (self.Base.graph_info[self.iblock][0], self.Base.graph_info[self.iblock][1])
        self.Base.cheml_type['transformer'].append(cheml_type)
        try:
            model = KFold(**self.parameters)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.SuperFunction) + type(
                err).__name__ + ': ' + err.message
            raise TypeError(msg)
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token == 'CV':
                self.Base.send[(self.iblock, token)] = [model, order.count(token)]
            else:
                msg = "@Task #%i(%s): non valid output token '%s'" % (self.iblock + 1, self.SuperFunction, token)
                raise NameError(msg)
        del self.legal_inputs

#####################################################################Regression

class OLS(BASE, Regressor):
    def legal_IO(self):
        self.legal_inputs = {'dfx': None, 'dfy': None}
        self.legal_outputs = {'r2_train': None, 'api': None}
        requirements = ['scikit_learn']
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def fit(self):
        from sklearn.linear_model import LinearRegression
        cheml_type = "%s_%s"%(self.Base.graph_info[self.iblock][0], self.Base.graph_info[self.iblock][1])
        self.Base.cheml_type['regressor'].append(cheml_type)
        try:
            model = LinearRegression(**self.parameters)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.SuperFunction) + type(
                err).__name__ + ': ' + err.message
            raise TypeError(msg)

        dfx = self.type_check('dfx', cheml_type='df', req=False, py_type=pd.DataFrame)
        dfy = self.type_check('dfy', cheml_type='df', req=False, py_type=pd.DataFrame)
        if not isinstance(dfx,type(None)) and not isinstance(dfy,type(None)):
            model, r2score_training = self.sklearn_training(model, dfx, dfy)

        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token == 'api':
                self.Base.send[(self.iblock, token)] = [model, order.count(token)]
            elif token == 'r2_train':
                if isinstance(dfx, type(None)) or isinstance(dfy, type(None)):
                    msg = "@Task #%i(%s): training needs both dfx and dfy" % (self.iblock + 1, self.SuperFunction)
                    raise NameError(msg)
                self.Base.send[(self.iblock, token)] = [r2score_training, order.count(token)]
            else:
                msg = "@Task #%i(%s): non valid output token '%s'" % (self.iblock + 1, self.SuperFunction, token)
                raise NameError(msg)
        del self.legal_inputs

class Ridge(BASE, Regressor):
    def legal_IO(self):
        self.legal_inputs = {'dfx': None, 'dfy': None}
        self.legal_outputs = {'r2_train': None, 'api': None}
        requirements = ['scikit_learn']
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def fit(self):
        from sklearn.linear_model import Ridge
        cheml_type = "%s_%s" % (self.Base.graph_info[self.iblock][0], self.Base.graph_info[self.iblock][1])
        self.Base.cheml_type['regressor'].append(cheml_type)
        try:
            model = Ridge(**self.parameters)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.SuperFunction) + type(
                err).__name__ + ': ' + err.message
            raise TypeError(msg)

        dfx = self.type_check('dfx', cheml_type='df', req=False, py_type=pd.DataFrame)
        dfy = self.type_check('dfy', cheml_type='df', req=False, py_type=pd.DataFrame)
        if not isinstance(dfx, type(None)) and not isinstance(dfy, type(None)):
            model, r2score_training = self.sklearn_training(model, dfx, dfy)

        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token == 'api':
                self.Base.send[(self.iblock, token)] = [model, order.count(token)]
            elif token == 'r2_train':
                if isinstance(dfx, type(None)) or isinstance(dfy, type(None)):
                    msg = "@Task #%i(%s): training needs both dfx and dfy" % (self.iblock + 1, self.SuperFunction)
                    raise NameError(msg)
                self.Base.send[(self.iblock, token)] = [r2score_training, order.count(token)]
            else:
                msg = "@Task #%i(%s): non valid output token '%s'" % (self.iblock + 1, self.SuperFunction, token)
                raise NameError(msg)
        del self.legal_inputs

class KernelRidge(BASE, Regressor):
    def legal_IO(self):
        self.legal_inputs = {'dfx': None, 'dfy': None}
        self.legal_outputs = {'r2_train': None, 'api': None}
        requirements = ['scikit_learn']
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def fit(self):
        from sklearn.kernel_ridge import KernelRidge
        cheml_type = "%s_%s" % (self.Base.graph_info[self.iblock][0], self.Base.graph_info[self.iblock][1])
        self.Base.cheml_type['regressor'].append(cheml_type)
        try:
            model = KernelRidge(**self.parameters)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.SuperFunction) + type(
                err).__name__ + ': ' + err.message
            raise TypeError(msg)

        dfx = self.type_check('dfx', cheml_type='df', req=False, py_type=pd.DataFrame)
        dfy = self.type_check('dfy', cheml_type='df', req=False, py_type=pd.DataFrame)
        if not isinstance(dfx, type(None)) and not isinstance(dfy, type(None)):
            model, r2score_training = self.sklearn_training(model, dfx, dfy)

        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token == 'api':
                self.Base.send[(self.iblock, token)] = [model, order.count(token)]
            elif token == 'r2_train':
                if isinstance(dfx, type(None)) or isinstance(dfy, type(None)):
                    msg = "@Task #%i(%s): training needs both dfx and dfy" % (self.iblock + 1, self.SuperFunction)
                    raise NameError(msg)
                self.Base.send[(self.iblock, token)] = [r2score_training, order.count(token)]
            else:
                msg = "@Task #%i(%s): non valid output token '%s'" % (self.iblock + 1, self.SuperFunction, token)
                raise NameError(msg)
        del self.legal_inputs

class Lasso(BASE, Regressor):
    def legal_IO(self):
        self.legal_inputs = {'dfx': None, 'dfy': None}
        self.legal_outputs = {'r2_train': None, 'api': None}
        requirements = ['scikit_learn']
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def fit(self):
        from sklearn.linear_model import Lasso
        cheml_type = "%s_%s" % (self.Base.graph_info[self.iblock][0], self.Base.graph_info[self.iblock][1])
        self.Base.cheml_type['regressor'].append(cheml_type)
        try:
            model = Lasso(**self.parameters)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.SuperFunction) + type(
                err).__name__ + ': ' + err.message
            raise TypeError(msg)

        dfx = self.type_check('dfx', cheml_type='df', req=False, py_type=pd.DataFrame)
        dfy = self.type_check('dfy', cheml_type='df', req=False, py_type=pd.DataFrame)
        if not isinstance(dfx, type(None)) and not isinstance(dfy, type(None)):
            model, r2score_training = self.sklearn_training(model, dfx, dfy)

        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token == 'api':
                self.Base.send[(self.iblock, token)] = [model, order.count(token)]
            elif token == 'r2_train':
                if isinstance(dfx, type(None)) or isinstance(dfy, type(None)):
                    msg = "@Task #%i(%s): training needs both dfx and dfy" % (self.iblock + 1, self.SuperFunction)
                    raise NameError(msg)
                self.Base.send[(self.iblock, token)] = [r2score_training, order.count(token)]
            else:
                msg = "@Task #%i(%s): non valid output token '%s'" % (self.iblock + 1, self.SuperFunction, token)
                raise NameError(msg)
        del self.legal_inputs

class MultiTaskLasso(BASE, Regressor):
    def legal_IO(self):
        self.legal_inputs = {'dfx': None, 'dfy': None}
        self.legal_outputs = {'r2_train': None, 'api': None}
        requirements = ['scikit_learn']
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def fit(self):
        from sklearn.linear_model import MultiTaskLasso
        cheml_type = "%s_%s" % (self.Base.graph_info[self.iblock][0], self.Base.graph_info[self.iblock][1])
        self.Base.cheml_type['regressor'].append(cheml_type)
        try:
            model = MultiTaskLasso(**self.parameters)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.SuperFunction) + type(
                err).__name__ + ': ' + err.message
            raise TypeError(msg)

        dfx = self.type_check('dfx', cheml_type='df', req=False, py_type=pd.DataFrame)
        dfy = self.type_check('dfy', cheml_type='df', req=False, py_type=pd.DataFrame)
        if not isinstance(dfx, type(None)) and not isinstance(dfy, type(None)):
            model, r2score_training = self.sklearn_training(model, dfx, dfy)

        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token == 'api':
                self.Base.send[(self.iblock, token)] = [model, order.count(token)]
            elif token == 'r2_train':
                if isinstance(dfx, type(None)) or isinstance(dfy, type(None)):
                    msg = "@Task #%i(%s): training needs both dfx and dfy" % (self.iblock + 1, self.SuperFunction)
                    raise NameError(msg)
                self.Base.send[(self.iblock, token)] = [r2score_training, order.count(token)]
            else:
                msg = "@Task #%i(%s): non valid output token '%s'" % (self.iblock + 1, self.SuperFunction, token)
                raise NameError(msg)
        del self.legal_inputs

class ElasticNet(BASE, Regressor):
    def legal_IO(self):
        self.legal_inputs = {'dfx': None, 'dfy': None}
        self.legal_outputs = {'r2_train': None, 'api': None}
        requirements = ['scikit_learn']
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def fit(self):
        from sklearn.linear_model import ElasticNet
        cheml_type = "%s_%s" % (self.Base.graph_info[self.iblock][0], self.Base.graph_info[self.iblock][1])
        self.Base.cheml_type['regressor'].append(cheml_type)
        try:
            model = ElasticNet(**self.parameters)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.SuperFunction) + type(
                err).__name__ + ': ' + err.message
            raise TypeError(msg)

        dfx = self.type_check('dfx', cheml_type='df', req=False, py_type=pd.DataFrame)
        dfy = self.type_check('dfy', cheml_type='df', req=False, py_type=pd.DataFrame)
        if not isinstance(dfx, type(None)) and not isinstance(dfy, type(None)):
            model, r2score_training = self.sklearn_training(model, dfx, dfy)

        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token == 'api':
                self.Base.send[(self.iblock, token)] = [model, order.count(token)]
            elif token == 'r2_train':
                if isinstance(dfx, type(None)) or isinstance(dfy, type(None)):
                    msg = "@Task #%i(%s): training needs both dfx and dfy" % (self.iblock + 1, self.SuperFunction)
                    raise NameError(msg)
                self.Base.send[(self.iblock, token)] = [r2score_training, order.count(token)]
            else:
                msg = "@Task #%i(%s): non valid output token '%s'" % (self.iblock + 1, self.SuperFunction, token)
                raise NameError(msg)
        del self.legal_inputs

class MultiTaskElasticNet(BASE, Regressor):
    def legal_IO(self):
        self.legal_inputs = {'dfx': None, 'dfy': None}
        self.legal_outputs = {'r2_train': None, 'api': None}
        requirements = ['scikit_learn']
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def fit(self):
        from sklearn.linear_model import MultiTaskElasticNet
        cheml_type = "%s_%s" % (self.Base.graph_info[self.iblock][0], self.Base.graph_info[self.iblock][1])
        self.Base.cheml_type['regressor'].append(cheml_type)
        try:
            model = MultiTaskElasticNet(**self.parameters)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.SuperFunction) + type(
                err).__name__ + ': ' + err.message
            raise TypeError(msg)

        dfx = self.type_check('dfx', cheml_type='df', req=False, py_type=pd.DataFrame)
        dfy = self.type_check('dfy', cheml_type='df', req=False, py_type=pd.DataFrame)
        if not isinstance(dfx, type(None)) and not isinstance(dfy, type(None)):
            model, r2score_training = self.sklearn_training(model, dfx, dfy)

        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token == 'api':
                self.Base.send[(self.iblock, token)] = [model, order.count(token)]
            elif token == 'r2_train':
                if isinstance(dfx, type(None)) or isinstance(dfy, type(None)):
                    msg = "@Task #%i(%s): training needs both dfx and dfy" % (self.iblock + 1, self.SuperFunction)
                    raise NameError(msg)
                self.Base.send[(self.iblock, token)] = [r2score_training, order.count(token)]
            else:
                msg = "@Task #%i(%s): non valid output token '%s'" % (self.iblock + 1, self.SuperFunction, token)
                raise NameError(msg)
        del self.legal_inputs

class Lars(BASE, Regressor):
    def legal_IO(self):
        self.legal_inputs = {'dfx': None, 'dfy': None}
        self.legal_outputs = {'r2_train': None, 'api': None}
        requirements = ['scikit_learn']
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def fit(self):
        from sklearn.linear_model import Lars
        cheml_type = "%s_%s" % (self.Base.graph_info[self.iblock][0], self.Base.graph_info[self.iblock][1])
        self.Base.cheml_type['regressor'].append(cheml_type)
        try:
            model = Lars(**self.parameters)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.SuperFunction) + type(
                err).__name__ + ': ' + err.message
            raise TypeError(msg)

        dfx = self.type_check('dfx', cheml_type='df', req=False, py_type=pd.DataFrame)
        dfy = self.type_check('dfy', cheml_type='df', req=False, py_type=pd.DataFrame)
        if not isinstance(dfx, type(None)) and not isinstance(dfy, type(None)):
            model, r2score_training = self.sklearn_training(model, dfx, dfy)

        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token == 'model':
                self.Base.send[(self.iblock, token)] = [model, order.count(token)]
            elif token == 'r2_train':
                if isinstance(dfx, type(None)) or isinstance(dfy, type(None)):
                    msg = "@Task #%i(%s): training needs both dfx and dfy" % (self.iblock + 1, self.SuperFunction)
                    raise NameError(msg)
                self.Base.send[(self.iblock, token)] = [r2score_training, order.count(token)]
            else:
                msg = "@Task #%i(%s): non valid output token '%s'" % (self.iblock + 1, self.SuperFunction, token)
                raise NameError(msg)
        del self.legal_inputs

class LassoLars(BASE, Regressor):
    def legal_IO(self):
        self.legal_inputs = {'dfx': None, 'dfy': None}
        self.legal_outputs = {'r2_train': None, 'api': None}
        requirements = ['scikit_learn']
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def fit(self):
        from sklearn.linear_model import LassoLars
        cheml_type = "%s_%s" % (self.Base.graph_info[self.iblock][0], self.Base.graph_info[self.iblock][1])
        self.Base.cheml_type['regressor'].append(cheml_type)
        try:
            model = LassoLars(**self.parameters)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.SuperFunction) + type(
                err).__name__ + ': ' + err.message
            raise TypeError(msg)

        dfx = self.type_check('dfx', cheml_type='df', req=False, py_type=pd.DataFrame)
        dfy = self.type_check('dfy', cheml_type='df', req=False, py_type=pd.DataFrame)
        if not isinstance(dfx, type(None)) and not isinstance(dfy, type(None)):
            model, r2score_training = self.sklearn_training(model, dfx, dfy)

        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token == 'model':
                self.Base.send[(self.iblock, token)] = [model, order.count(token)]
            elif token == 'r2_train':
                if isinstance(dfx, type(None)) or isinstance(dfy, type(None)):
                    msg = "@Task #%i(%s): training needs both dfx and dfy" % (self.iblock + 1, self.SuperFunction)
                    raise NameError(msg)
                self.Base.send[(self.iblock, token)] = [r2score_training, order.count(token)]
            else:
                msg = "@Task #%i(%s): non valid output token '%s'" % (self.iblock + 1, self.SuperFunction, token)
                raise NameError(msg)
        del self.legal_inputs

class BayesianRidge(BASE, Regressor):
    def legal_IO(self):
        self.legal_inputs = {'dfx': None, 'dfy': None}
        self.legal_outputs = {'r2_train': None, 'api': None}
        requirements = ['scikit_learn']
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def fit(self):
        from sklearn.linear_model import BayesianRidge
        cheml_type = "%s_%s" % (self.Base.graph_info[self.iblock][0], self.Base.graph_info[self.iblock][1])
        self.Base.cheml_type['regressor'].append(cheml_type)
        try:
            model = BayesianRidge(**self.parameters)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.SuperFunction) + type(
                err).__name__ + ': ' + err.message
            raise TypeError(msg)

        dfx = self.type_check('dfx', cheml_type='df', req=False, py_type=pd.DataFrame)
        dfy = self.type_check('dfy', cheml_type='df', req=False, py_type=pd.DataFrame)
        if not isinstance(dfx, type(None)) and not isinstance(dfy, type(None)):
            model, r2score_training = self.sklearn_training(model, dfx, dfy)

        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token == 'model':
                self.Base.send[(self.iblock, token)] = [model, order.count(token)]
            elif token == 'r2_train':
                if isinstance(dfx, type(None)) or isinstance(dfy, type(None)):
                    msg = "@Task #%i(%s): training needs both dfx and dfy" % (self.iblock + 1, self.SuperFunction)
                    raise NameError(msg)
                self.Base.send[(self.iblock, token)] = [r2score_training, order.count(token)]
            else:
                msg = "@Task #%i(%s): non valid output token '%s'" % (self.iblock + 1, self.SuperFunction, token)
                raise NameError(msg)
        del self.legal_inputs

class ARD(BASE, Regressor):
    def legal_IO(self):
        self.legal_inputs = {'dfx': None, 'dfy': None}
        self.legal_outputs = {'r2_train': None, 'api': None}
        requirements = ['scikit_learn']
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def fit(self):
        from sklearn.linear_model import ARDRegression
        cheml_type = "%s_%s" % (self.Base.graph_info[self.iblock][0], self.Base.graph_info[self.iblock][1])
        self.Base.cheml_type['regressor'].append(cheml_type)
        try:
            model = ARDRegression(**self.parameters)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.SuperFunction) + type(
                err).__name__ + ': ' + err.message
            raise TypeError(msg)

        dfx = self.type_check('dfx', cheml_type='df', req=False, py_type=pd.DataFrame)
        dfy = self.type_check('dfy', cheml_type='df', req=False, py_type=pd.DataFrame)
        if not isinstance(dfx, type(None)) and not isinstance(dfy, type(None)):
            model, r2score_training = self.sklearn_training(model, dfx, dfy)

        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token == 'model':
                self.Base.send[(self.iblock, token)] = [model, order.count(token)]
            elif token == 'r2_train':
                if isinstance(dfx, type(None)) or isinstance(dfy, type(None)):
                    msg = "@Task #%i(%s): training needs both dfx and dfy" % (self.iblock + 1, self.SuperFunction)
                    raise NameError(msg)
                self.Base.send[(self.iblock, token)] = [r2score_training, order.count(token)]
            else:
                msg = "@Task #%i(%s): non valid output token '%s'" % (self.iblock + 1, self.SuperFunction, token)
                raise NameError(msg)
        del self.legal_inputs

class Logistic(BASE, Regressor):
    def legal_IO(self):
        self.legal_inputs = {'dfx': None, 'dfy': None}
        self.legal_outputs = {'r2_train': None, 'api': None}
        requirements = ['scikit_learn']
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def fit(self):
        from sklearn.linear_model import LogisticRegression
        cheml_type = "%s_%s" % (self.Base.graph_info[self.iblock][0], self.Base.graph_info[self.iblock][1])
        self.Base.cheml_type['regressor'].append(cheml_type)
        try:
            model = LogisticRegression(**self.parameters)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.SuperFunction) + type(
                err).__name__ + ': ' + err.message
            raise TypeError(msg)

        dfx = self.type_check('dfx', cheml_type='df', req=False, py_type=pd.DataFrame)
        dfy = self.type_check('dfy', cheml_type='df', req=False, py_type=pd.DataFrame)
        if not isinstance(dfx, type(None)) and not isinstance(dfy, type(None)):
            model, r2score_training = self.sklearn_training(model, dfx, dfy)

        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token == 'model':
                self.Base.send[(self.iblock, token)] = [model, order.count(token)]
            elif token == 'r2_train':
                if isinstance(dfx, type(None)) or isinstance(dfy, type(None)):
                    msg = "@Task #%i(%s): training needs both dfx and dfy" % (self.iblock + 1, self.SuperFunction)
                    raise NameError(msg)
                self.Base.send[(self.iblock, token)] = [r2score_training, order.count(token)]
            else:
                msg = "@Task #%i(%s): non valid output token '%s'" % (self.iblock + 1, self.SuperFunction, token)
                raise NameError(msg)
        del self.legal_inputs

class SGD(BASE, Regressor):
    def legal_IO(self):
        self.legal_inputs = {'dfx': None, 'dfy': None}
        self.legal_outputs = {'r2_train': None, 'api': None}
        requirements = ['scikit_learn']
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def fit(self):
        from sklearn.linear_model import SGDRegressor
        cheml_type = "%s_%s" % (self.Base.graph_info[self.iblock][0], self.Base.graph_info[self.iblock][1])
        self.Base.cheml_type['regressor'].append(cheml_type)
        try:
            model = SGDRegressor(**self.parameters)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.SuperFunction) + type(
                err).__name__ + ': ' + err.message
            raise TypeError(msg)

        dfx = self.type_check('dfx', cheml_type='df', req=False, py_type=pd.DataFrame)
        dfy = self.type_check('dfy', cheml_type='df', req=False, py_type=pd.DataFrame)
        if not isinstance(dfx, type(None)) and not isinstance(dfy, type(None)):
            model, r2score_training = self.sklearn_training(model, dfx, dfy)

        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token == 'model':
                self.Base.send[(self.iblock, token)] = [model, order.count(token)]
            elif token == 'r2_train':
                if isinstance(dfx, type(None)) or isinstance(dfy, type(None)):
                    msg = "@Task #%i(%s): training needs both dfx and dfy" % (self.iblock + 1, self.SuperFunction)
                    raise NameError(msg)
                self.Base.send[(self.iblock, token)] = [r2score_training, order.count(token)]
            else:
                msg = "@Task #%i(%s): non valid output token '%s'" % (self.iblock + 1, self.SuperFunction, token)
                raise NameError(msg)
        del self.legal_inputs

class SVR(BASE, Regressor):
    def legal_IO(self):
        self.legal_inputs = {'dfx': None, 'dfy': None}
        self.legal_outputs = {'r2_train': None, 'model': None}
        requirements = ['scikit_learn']
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def fit(self):
        # step1: check inputs
        dfx, dfx_info = self.input_check('dfx', req = True, py_type = pd.DataFrame)
        dfy, dfy_info = self.input_check('dfy', req = True, py_type = pd.DataFrame)

        from sklearn.svm import SVR

        try:
            model = SVR(**self.parameters)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.SuperFunction) + type(
                err).__name__ + ': ' + err.message
            raise TypeError(msg)

        dfx = self.type_check('dfx', cheml_type='df', req=False, py_type=pd.DataFrame)
        dfy = self.type_check('dfy', cheml_type='df', req=False, py_type=pd.DataFrame)
        if dfy is not None and dfy.shape[1]==1:
            dfy = np.ravel(dfy)
        if not isinstance(dfx, type(None)) and not isinstance(dfy, type(None)):
            model, r2score_training = self.sklearn_training(model, dfx, dfy)

        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token == 'model':
                self.Base.send[(self.iblock, token)] = [model, order.count(token)]
            elif token == 'r2_train':
                if isinstance(dfx, type(None)) or isinstance(dfy, type(None)):
                    msg = "@Task #%i(%s): training needs both dfx and dfy" % (self.iblock + 1, self.SuperFunction)
                    raise NameError(msg)
                self.Base.send[(self.iblock, token)] = [r2score_training, order.count(token)]
            else:
                msg = "@Task #%i(%s): non valid output token '%s'" % (self.iblock + 1, self.SuperFunction, token)
                raise NameError(msg)
        del self.legal_inputs

class NuSVR(BASE, Regressor):
    def legal_IO(self):
        self.legal_inputs = {'dfx': None, 'dfy': None}
        self.legal_outputs = {'r2_train': None, 'model': None}
        requirements = ['scikit_learn']
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def fit(self):
        from sklearn.svm import NuSVR
        cheml_type = "%s_%s" % (self.Base.graph_info[self.iblock][0], self.Base.graph_info[self.iblock][1])
        self.Base.cheml_type['regressor'].append(cheml_type)
        try:
            model = NuSVR(**self.parameters)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.SuperFunction) + type(
                err).__name__ + ': ' + err.message
            raise TypeError(msg)

        dfx = self.type_check('dfx', cheml_type='df', req=False, py_type=pd.DataFrame)
        dfy = self.type_check('dfy', cheml_type='df', req=False, py_type=pd.DataFrame)
        if not isinstance(dfx, type(None)) and not isinstance(dfy, type(None)):
            model, r2score_training = self.sklearn_training(model, dfx, dfy)

        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token == 'model':
                self.Base.send[(self.iblock, token)] = [model, order.count(token)]
            elif token == 'r2_train':
                if isinstance(dfx, type(None)) or isinstance(dfy, type(None)):
                    msg = "@Task #%i(%s): training needs both dfx and dfy" % (self.iblock + 1, self.SuperFunction)
                    raise NameError(msg)
                self.Base.send[(self.iblock, token)] = [r2score_training, order.count(token)]
            else:
                msg = "@Task #%i(%s): non valid output token '%s'" % (self.iblock + 1, self.SuperFunction, token)
                raise NameError(msg)
        del self.legal_inputs

class LinearSVR(BASE, Regressor):
    def legal_IO(self):
        self.legal_inputs = {'dfx': None, 'dfy': None}
        self.legal_outputs = {'r2_train': None, 'model': None}
        requirements = ['scikit_learn']
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def fit(self):
        from sklearn.svm import LinearSVR
        cheml_type = "%s_%s" % (self.Base.graph_info[self.iblock][0], self.Base.graph_info[self.iblock][1])
        self.Base.cheml_type['regressor'].append(cheml_type)
        try:
            model = LinearSVR(**self.parameters)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.SuperFunction) + type(
                err).__name__ + ': ' + err.message
            raise TypeError(msg)

        dfx = self.type_check('dfx', cheml_type='df', req=False, py_type=pd.DataFrame)
        dfy = self.type_check('dfy', cheml_type='df', req=False, py_type=pd.DataFrame)
        if not isinstance(dfx, type(None)) and not isinstance(dfy, type(None)):
            model, r2score_training = self.sklearn_training(model, dfx, dfy)

        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token == 'model':
                self.Base.send[(self.iblock, token)] = [model, order.count(token)]
            elif token == 'r2_train':
                if isinstance(dfx, type(None)) or isinstance(dfy, type(None)):
                    msg = "@Task #%i(%s): training needs both dfx and dfy" % (self.iblock + 1, self.SuperFunction)
                    raise NameError(msg)
                self.Base.send[(self.iblock, token)] = [r2score_training, order.count(token)]
            else:
                msg = "@Task #%i(%s): non valid output token '%s'" % (self.iblock + 1, self.SuperFunction, token)
                raise NameError(msg)
        del self.legal_inputs

#####################################################################Postprocessor

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
                self.Base.send[(self.iblock, token)] = [best_estimator_, order.count(token)]
            elif token == 'cv_results_':
                self.Base.send[(self.iblock, token)] = [pd.DataFrame(api.cv_results_), order.count(token)]
            elif token == 'api':
                self.Base.send[(self.iblock, token)] = [api, order.count(token)]
            else:
                msg = "@Task #%i(%s): non valid output token '%s'" % (self.iblock+1, self.SuperFunction, token)
                raise NameError(msg)

        #step7: delete all inputs from memory
        del self.legal_inputs

class Evaluation(BASE, Regressor, Evaluator):
    def legal_IO(self):
        self.legal_inputs = {'dfx': None, 'dfy': None, 'CV': None, 'X_scaler': None, 'Y_scaler': None, 'model': None}
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
        cheml_type = "%s_%s" % (self.Base.graph_info[self.iblock][0], self.Base.graph_info[self.iblock][1])
        self.Base.cheml_type['evaluator'].append(cheml_type)
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
