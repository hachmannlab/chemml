import warnings
import pandas as pd
import numpy as np
import copy


from .syntax import Preprocessor, Regressor, Evaluator
from chemml.wrapper.base import BASE
from chemml.utils import value


##################################################################### 2 Prepare Data

class automatic_run(BASE):
    def fit(self):
        self.Fit_sklearn()
        # tokens = ['df','dfy_predict']
        # for token in tokens:
        #     if token in self.outputs:
        #         if self.outputs[token].value is not None:
        #             if self.header:
        #                 df = self.outputs[token].value
        #                 df = pd.DataFrame(df, columns=self.inputs[token[0:3]].value.columns)
        #                 self.outputs[token].value = df
        #             else:
        #                 self.outputs[token].value = pd.DataFrame(self.outputs[token].value)
        self.Send()
        # delete all inputs
        del self.inputs


# feature representation
class PolynomialFeatures(BASE):
    def fit(self):
        self.Fit_sklearn()
        # headers from class method: get_feature_names
        if self.method in ['fit_transform','transform']:
            if self.header:
                df = self.outputs['df'].value
                df = pd.DataFrame(df, columns = self.outputs['api'].value.get_feature_names())
                self.outputs['df'].value = df
            else:
                df = self.outputs['df'].value
                df = pd.DataFrame(df)
                self.outputs['df'].value = df

        self.Send()
        # delete all inputs
        del self.inputs


class OneHotEncoder(BASE):
    def fit(self):
        self.Fit_sklearn()
        # .toarray() is requied
        if self.method in ['fit_transform','transform']:
            df = self.outputs['df'].value.toarray()
            df = pd.DataFrame(df, columns = self.inputs['df'].value.columns)
            self.outputs['df'].value = df
        self.Send()
        # delete all inputs
        del self.inputs


# basic oprator

class Imputer(BASE):
    def fit(self):
        self.Fit_sklearn()
        # headers from input df and based on the class attribute: statistics_
        if self.method in ['fit_transform', 'transform']:
            if self.header:
                df = self.outputs['df'].value
                api = self.outputs['api'].value
                stats = api.statistics_
                NOTnan_ind = [i for i, val in enumerate(stats) if not np.isnan(val)]
                df = pd.DataFrame(df, columns=self.inputs['df'].value.columns[NOTnan_ind])
                self.outputs['df'].value = df
            else:
                df = self.outputs['df'].value
                df = pd.DataFrame(df)
                self.outputs['df'].value = df
        self.Send()
        # delete all inputs
        del self.inputs

# scaler

# feature selector

# feature transformer


# splitter

class train_test_split(BASE):
    def fit(self):
        # step1: check inputs
        self.required('dfx', req=True)
        dfx = self.inputs['dfx'].value
        dfy = self.inputs['dfy'].value

        # step2: assign inputs to parameters if necessary (param = @token)
        self.paramFROMinput()

        # step3: check the dimension of input data frame
        dfx, _ = self.data_check('dfx', dfx, ndim=2, n0=None, n1=None, format_out='df')
        # step4: import module and make APIs
        try:
            exec ("from %s.%s import %s" % (self.metadata.modules[0], self.metadata.modules[1], self.Function))
            submodule = getattr(__import__(self.metadata.modules[0]), self.metadata.modules[1])
            F = getattr(submodule, self.Function)
            if dfy is None:
                tts_out = F(dfx, **self.parameters)
                dfx_train, dfx_test = tts_out
                self.set_value('dfx_train', pd.DataFrame(dfx_train,columns=dfx.columns))
                self.set_value('dfx_test',pd.DataFrame(dfx_test,columns=dfx.columns))
            else:
                dfy, _ = self.data_check('dfy', dfy, ndim=1, n0=dfx.shape[0], n1=None, format_out='df')
                tts_out = F(dfx, dfy, **self.parameters)
                dfx_train, dfx_test, dfy_train, dfy_test = tts_out
                self.set_value('dfx_train', pd.DataFrame(dfx_train,columns=dfx.columns))
                self.set_value('dfx_test', pd.DataFrame(dfx_test,columns=dfx.columns))
                self.set_value('dfy_train', pd.DataFrame(dfy_train,columns=dfy.columns))
                self.set_value('dfy_test', pd.DataFrame(dfy_test,columns=dfy.columns))
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(
                err).__name__ + ': ' + err.message
            raise TypeError(msg)


        # send and delete inputs
        self.Send()
        del self.inputs
        del dfx
        del dfy

class KFold(BASE):
    def fit(self):
        self.paramFROMinput()
        if 'func_method' in self.parameters:
            method = self.parameters.pop('func_method')
        else:
            method = None
        if method not in self.metadata.WParameters.func_method.options:
            msg = "@Task #%i(%s): The method '%s' is not available for the function '%s'." % (
                self.iblock, self.Task,method,self.Function)
            raise NameError(msg)
        else:
            if method == None:
                try:
                    exec ("from %s.%s import %s" % (self.metadata.modules[0], self.metadata.modules[1], self.Function))
                    submodule = getattr(__import__(self.metadata.modules[0]), self.metadata.modules[1])
                    F = getattr(submodule, self.Function)
                    api = F(**self.parameters)
                except Exception as err:
                    msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(err).__name__ + ': ' + err.message
                    raise TypeError(msg)
                self.set_value('api', api)
            elif method == 'split':
                try:
                    exec ("from %s.%s import %s" % (self.metadata.modules[0], self.metadata.modules[1], self.Function))
                    submodule = getattr(__import__(self.metadata.modules[0]), self.metadata.modules[1])
                    F = getattr(submodule, self.Function)
                    api = F(**self.parameters)
                except Exception as err:
                    msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(err).__name__ + ': ' + err.message
                    raise TypeError(msg)
                self.required('dfx', req=True)
                dfx = self.inputs['dfx'].value
                fold_gen = api.split(dfx)
                self.set_value('api', api)
                self.set_value('fold_gen', fold_gen)

        self.Send()
        del self.inputs

class LeaveOneOut(BASE):
    def fit(self):
        self.paramFROMinput()
        if 'func_method' in self.parameters:
            method = self.parameters.pop('func_method')
        else:
            method = None
        if method not in self.metadata.WParameters.func_method.options:
            msg = "@Task #%i(%s): The method '%s' is not available for the function '%s'." % (
                self.iblock, self.Task,method,self.Function)
            raise NameError(msg)
        else:
            if method == None:
                try:
                    exec ("from %s.%s import %s" % (self.metadata.modules[0], self.metadata.modules[1], self.Function))
                    submodule = getattr(__import__(self.metadata.modules[0]), self.metadata.modules[1])
                    F = getattr(submodule, self.Function)
                    api = F(**self.parameters)
                except Exception as err:
                    msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(err).__name__ + ': ' + err.message
                    raise TypeError(msg)
                self.set_value('api', api)
            elif method == 'split':
                try:
                    exec ("from %s.%s import %s" % (self.metadata.modules[0], self.metadata.modules[1], self.Function))
                    submodule = getattr(__import__(self.metadata.modules[0]), self.metadata.modules[1])
                    F = getattr(submodule, self.Function)
                    api = F(**self.parameters)
                except Exception as err:
                    msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(err).__name__ + ': ' + err.message
                    raise TypeError(msg)
                self.required('dfx', req=True)
                dfx = self.inputs['dfx'].value
                fold_gen = api.split(dfx)
                self.set_value('api', api)
                self.set_value('fold_gen', fold_gen)

        self.Send()
        del self.inputs

class ShuffleSplit(BASE):
    def fit(self):
        self.paramFROMinput()
        if 'func_method' in self.parameters:
            method = self.parameters.pop('func_method')
        else:
            method = None
        if method not in self.metadata.WParameters.func_method.options:
            msg = "@Task #%i(%s): The method '%s' is not available for the function '%s'." % (
                self.iblock, self.Task,method,self.Function)
            raise NameError(msg)
        else:
            if method == None:
                try:
                    exec ("from %s.%s import %s" % (self.metadata.modules[0], self.metadata.modules[1], self.Function))
                    submodule = getattr(__import__(self.metadata.modules[0]), self.metadata.modules[1])
                    F = getattr(submodule, self.Function)
                    api = F(**self.parameters)
                except Exception as err:
                    msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(err).__name__ + ': ' + err.message
                    raise TypeError(msg)
                self.set_value('api', api)
            elif method == 'split':
                try:
                    exec ("from %s.%s import %s" % (self.metadata.modules[0], self.metadata.modules[1], self.Function))
                    submodule = getattr(__import__(self.metadata.modules[0]), self.metadata.modules[1])
                    F = getattr(submodule, self.Function)
                    api = F(**self.parameters)
                except Exception as err:
                    msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(err).__name__ + ': ' + err.message
                    raise TypeError(msg)
                self.required('dfx', req=True)
                dfx = self.inputs['dfx'].value
                fold_gen = api.split(dfx)
                self.set_value('api', api)
                self.set_value('fold_gen', fold_gen)

        self.Send()
        del self.inputs

class StratifiedShuffleSplit(BASE):
    def fit(self):
        self.paramFROMinput()
        if 'func_method' in self.parameters:
            method = self.parameters.pop('func_method')
        else:
            method = None
        if method not in self.metadata.WParameters.func_method.options:
            msg = "@Task #%i(%s): The method '%s' is not available for the function '%s'." % (
                self.iblock, self.Task,method,self.Function)
            raise NameError(msg)
        else:
            if method == None:
                try:
                    exec ("from %s.%s import %s" % (self.metadata.modules[0], self.metadata.modules[1], self.Function))
                    submodule = getattr(__import__(self.metadata.modules[0]), self.metadata.modules[1])
                    F = getattr(submodule, self.Function)
                    api = F(**self.parameters)
                except Exception as err:
                    msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(err).__name__ + ': ' + err.message
                    raise TypeError(msg)
                self.set_value('api', api)
            elif method == 'split':
                try:
                    exec ("from %s.%s import %s" % (self.metadata.modules[0], self.metadata.modules[1], self.Function))
                    submodule = getattr(__import__(self.metadata.modules[0]), self.metadata.modules[1])
                    F = getattr(submodule, self.Function)
                    api = F(**self.parameters)
                except Exception as err:
                    msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(err).__name__ + ': ' + err.message
                    raise TypeError(msg)
                self.required('dfx', req=True)
                dfx = self.inputs['dfx'].value
                fold_gen = api.split(dfx)
                self.set_value('api', api)
                self.set_value('fold_gen', fold_gen)

        self.Send()
        del self.inputs


##################################################################### 3 Define Model

# Regression


##################################################################### 4 Search

class GridSearchCV(BASE):
    def fit(self):
        # step1: check inputs
        self.required('dfx', req=True)
        self.required('estimator', req=True)
        dfx = self.inputs['dfx'].value
        dfy = self.inputs['dfy'].value
        estimator = self.inputs['estimator'].value
        self.parameters['estimator'] = estimator

        # step2: check parameters
        # Note: estimator is a required parameter and can be received from the input stream
        # Note: scorer is not a required parameter but can be received from the input stream
        self.paramFROMinput()

        # step3: check the dimension of input data frame
        dfx, _ = self.data_check('dfx', dfx, ndim=2, n0=None, n1=None, format_out='ar')
        if dfy is not None:
            dfy, _ = self.data_check('dfy', dfy, ndim=1, n0=dfx.shape[0], n1=None, format_out='ar')

        print (dfx.shape)
        print (dfy.shape)

        # step4: import module and make API
        try:
            exec ("from %s.%s import %s" % (self.metadata.modules[0], self.metadata.modules[1], self.Function))
            submodule = getattr(__import__(self.metadata.modules[0]), self.metadata.modules[1])
            F = getattr(submodule, self.Function)
            api = F(**self.parameters)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(
                err).__name__ + ': ' + err.message
            raise TypeError(msg)

        # step5: process
        api.fit(dfx, dfy)

        # step6: send
        order = [edge[1] for edge in self.Base.graph if edge[0]==self.iblock]
        for token in set(order):
            if token not in self.outputs:
                msg = "@Task #%i(%s): not a valid output token '%s'" % (self.iblock + 1, self.Task, token)
                raise NameError(msg)
            elif token == 'best_estimator_':
                if api.get_params()['refit']:
                    best_estimator_ = copy.deepcopy(api.best_estimator_)
                else:
                    best_estimator_ = copy.deepcopy(self.parameters['estimator'])
                    best_estimator_.set_params(**api.best_params_)
                    # best_estimator_.fit(dfx,dfy)
                self.set_value('best_estimator_', best_estimator_)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]
            elif token == 'cv_results_':
                self.set_value('cv_results_', pd.DataFrame(api.cv_results_))
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]
            elif token == 'api':
                self.set_value('api',api)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]

        # step7: delete inputs
        del self.inputs
        del dfx
        del dfy

class cross_val_score(BASE):
    def fit(self):
        # step1: check inputs
        self.required('dfx', req=True)
        self.required('estimator', req=True)
        dfx = self.inputs['dfx'].value
        dfy = self.inputs['dfy'].value
        estimator = self.inputs['estimator'].value

        # step2: check parameters
        # Note: estimator is a required parameter and can be received from the input stream
        # Note: scorer is not a required parameter but can be received from the input stream
        self.paramFROMinput()
        if 'estimator' not in self.parameters:
            self.parameters['estimator'] = estimator
        if 'X' not in self.parameters:
            self.parameters['X'] = dfx


        # step3: check the dimension of input data frame
        dfx, _ = self.data_check('dfx', dfx, ndim=2, n0=None, n1=None, format_out='ar')
        if dfy is not None:
            dfy, _ = self.data_check('dfy', dfy, ndim=1, n0=dfx.shape[0], n1=None, format_out='ar')

        # step4: import module and make APIs
        try:
            exec ("from %s.%s import %s" % (self.metadata.modules[0], self.metadata.modules[1], self.Function))
            submodule = getattr(__import__(self.metadata.modules[0]), self.metadata.modules[1])
            F = getattr(submodule, self.Function)
            scores = F(**self.parameters)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(
                err).__name__ + ': ' + err.message
            raise TypeError(msg)

        # step6: set outputs
        order = [edge[1] for edge in self.Base.graph if edge[0]==self.iblock]
        for token in set(order):
            if token == 'scores':
                self.set_value(token, pd.DataFrame(scores))
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]
            else:
                msg = "@Task #%i(%s): not a valid output token '%s'" % (self.iblock + 1, self.Task, token)
                raise NameError(msg)

        # step7: delete all inputs from memory
        del self.inputs
        del dfx
        del dfy

class cross_val_predict(BASE):
    def fit(self):
        # step1: check inputs
        self.required('dfx', req=True)
        self.required('estimator', req=True)
        dfx = self.inputs['dfx'].value
        dfy = self.inputs['dfy'].value
        estimator = self.inputs['estimator'].value
        self.parameters['estimator'] = estimator
        # step2: check parameters
        # Note: estimator is a required parameter and can be received from the input stream
        # Note: scorer is not a required parameter but can be received from the input stream
        self.paramFROMinput()

        # step3: check the dimension of input data frame
        dfx, _ = self.data_check('dfx', dfx, ndim=2, n0=None, n1=None, format_out='ar')
        if dfy is not None:
            dfy, _ = self.data_check('dfy', dfy, ndim=1, n0=dfx.shape[0], n1=None, format_out='ar')

        # step4: import module and make APIs
        try:
            exec ("from %s.%s import %s" % (self.metadata.modules[0], self.metadata.modules[1], self.Function))
            submodule = getattr(__import__(self.metadata.modules[0]), self.metadata.modules[1])
            F = getattr(submodule, self.Function)
            dfy_pred = F(**self.parameters)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(
                err).__name__ + ': ' + err.message
            raise TypeError(msg)

        # step6: set outputs
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token == 'dfy_predict':
                self.set_value(token, pd.DataFrame(dfy_pred))
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]
            else:
                msg = "@Task #%i(%s): not a valid output token '%s'" % (self.iblock + 1, self.Task, token)
                raise NameError(msg)

        # step7: delete all inputs from memory
        del self.inputs
        del dfx
        del dfy

class learning_curve(BASE):
    def fit(self):
        # step1: check inputs
        self.required('dfx', req=True)
        self.required('estimator', req=True)
        dfx = self.inputs['dfx'].value
        dfy = self.inputs['dfy'].value
        estimator = self.inputs['estimator'].value
        self.parameters['estimator'] = estimator

        # step2: check parameters
        # Note: estimator is a required parameter and can be received from the input stream
        # Note: scorer is not a required parameter but can be received from the input stream
        self.paramFROMinput()
        if type(self.parameters['train_sizes']) is str:
            self.parameters['train_sizes'] = eval(self.parameters['train_sizes'])

        # step3: check the dimension of input data frame
        dfx, _ = self.data_check('dfx', dfx, ndim=2, n0=None, n1=None, format_out='ar')
        if dfy is not None:
            dfy, _ = self.data_check('dfy', dfy, ndim=1, n0=dfx.shape[0], n1=None, format_out='ar')

        # step4: import module and make APIs
        try:
            exec ("from %s.%s import %s" % (self.metadata.modules[0], self.metadata.modules[1], self.Function))
            submodule = getattr(__import__(self.metadata.modules[0]), self.metadata.modules[1])
            F = getattr(submodule, self.Function)
            train_sizes_abs, train_scores, test_scores = F(X=dfx,y=dfy,**self.parameters)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(
                err).__name__ + ': ' + err.message
            raise TypeError(msg)

        # step6: send out
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token == 'train_sizes_abs':
                self.set_value(token, train_sizes_abs)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]
            elif token == 'train_scores':
                self.set_value(token, train_scores)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]
            elif token == 'test_scores':
                self.set_value(token, test_scores)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]
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

                self.set_value(token, extended_result_)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]
            else:
                msg = "@Task #%i(%s): non valid output token '%s'" % (self.iblock + 1, self.Task, token)
                raise NameError(msg)

        # step7: delete all inputs from memory
        del self.inputs


# Metrics

class evaluate_regression(BASE,Evaluator):
    def fit(self):
        # step2: assign inputs to parameters if necessary (param = @token)
        self.paramFROMinput()

        # step4: import module and make APIs
        try:
            self._reg_evaluation_params()
        except Exception as err:
            msg = '@Task #%i(%s): '%(self.iblock+1, self.Task) + type(err).__name__ + ': '+ err.message
            raise TypeError(msg)

        # step5: process
        # step6: send out
        order = [edge[1] for edge in self.Base.graph if edge[0]==self.iblock]
        for token in set(order):
            if token == 'evaluator':
                self.set_value(token, self.evaluator)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]
            elif token == 'evaluation_results_':
                # step1: check inputs
                self.required('dfy', req=True)
                self.required('dfy_predict', req=True)
                dfy = self.inputs['dfy'].value
                dfy_predict = self.inputs['dfy_predict'].value

                # step3: check the dimension of input data frame
                dfy, _ = self.data_check('dfy', dfy, ndim=2, n0=None, n1=None, format_out='df')
                dfy_predict, _ = self.data_check('dfy_predict', dfy_predict, ndim=2, n0=dfy.shape[0], n1=None, format_out='df')

                self._reg_evaluate(dfy, dfy_predict, self.evaluator)
                evaluation_results_ = pd.DataFrame(self.results)
                self.set_value(token, evaluation_results_)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]
            else:
                msg = "@Task #%i(%s): non valid output token '%s'" % (self.iblock+1, self.Task, token)
                raise NameError(msg)

        #step7: delete all inputs from memory
        del self.inputs

class scorer_regression(BASE):
    def fit(self):
        # step1: check inputs
        # step2: assign inputs to parameters if necessary (param = @token)
        self.paramFROMinput()
        if 'metric' in self.parameters:
            metric = self.parameters.pop('metric')
        else:
            metric = 'mae'
        if 'kwargs' in self.parameters:
            kwargs = self.parameters.pop('kwargs')
            for item in kwargs:
                self.parameters[item] = value(kwargs[item])

        # Todo: add all the metrics for regression

        # step3: check the dimension of input data frame
        # step4: import module and make APIs
        try:
            exec ("from %s.%s import %s" % (self.metadata.modules[0], self.metadata.modules[1], 'make_scorer'))
            submodule = getattr(__import__(self.metadata.modules[0]), self.metadata.modules[1])
            F = getattr(submodule, 'make_scorer')
            if metric == 'mae':
                from sklearn.metrics import mean_absolute_error
                scorer = F(mean_absolute_error,**self.parameters)
            elif metric =='r2':
                from sklearn.metrics import r2_score
                scorer = F(r2_score,**self.parameters)
            elif metric == 'mse':
                from sklearn.metrics import mean_squared_error
                scorer = F(mean_squared_error, **self.parameters)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(
                err).__name__ + ': ' + err.message
            raise TypeError(msg)
        # step5: process
        # step6: send out
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token == 'scorer':
                self.set_value(token, scorer)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]
            else:
                msg = "@Task #%i(%s): non valid output token '%s'" % (self.iblock + 1, self.Task, token)
                raise NameError(msg)
        # step7: delete all inputs from memory
        del self.inputs


##################################################################### 5 Mix
