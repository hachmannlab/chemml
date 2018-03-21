import copy
import warnings

import numpy as np
import pandas as pd

from .database import sklearn_db
from .database import cheml_db
from .database import pandas_db


# todo: decorate some of the steps in the wrapeprs. e.g. sending out ouputs by finding all the connected edges in the graph
# todo: use Input and Output classes to handle inputs and outputs

class BASE(object):
    """
    Do not instantiate this class
    """
    def __init__(self, Base, parameters, iblock, Task, Function, Host):
        self.Base = Base
        self.parameters = parameters
        self.iblock = iblock
        self.Task = Task
        self.Function = Function
        self.Host = Host

    def run(self):
        self.IO()
        self.Receive()
        self.fit()

    def IO(self):
        if self.Host == 'sklearn':
            self.metadata = getattr(sklearn_db, self.Function)()
        elif self.Host == 'cheml':
            self.metadata = getattr(cheml_db, self.Function)()
        elif self.Host == 'pandas':
            self.metadata = getattr(pandas_db, self.Function)()
        self.inputs = {i:copy.deepcopy(vars(self.metadata.Inputs)[i]) for i in vars(self.metadata.Inputs).keys() if
                       i not in ('__module__','__doc__')}
        self.outputs = {i:copy.deepcopy(vars(self.metadata.Outputs)[i]) for i in vars(self.metadata.Outputs).keys() if
                   i not in ('__module__', '__doc__')}
        # self.wparams = {i:copy.deepcopy(vars(self.metadata.WParameters)[i]) for i in vars(self.metadata.WParameters).keys() if
        #             i not in ('__module__', '__doc__')}

    def Receive(self):
        recv = [edge for edge in self.Base.graph if edge[2] == self.iblock]
        self.Base.graph = tuple([edge for edge in self.Base.graph if edge[2] != self.iblock])
        # check received tokens to: (1) be a legal input, and (2) be unique.
        count = {token: 0 for token in self.inputs}
        for edge in recv:
            if edge[3] in self.inputs:
                count[edge[3]] += 1
                if count[edge[3]] > 1:
                    msg = '@Task #%i(%s): only one input per each available input token can be received.' % (
                        self.iblock + 1, self.Task)
                    raise IOError(msg)
            else:
                msg = "@Task #%i(%s): received a non valid input token '%s', sent by block #%i" % (
                    self.iblock + 1, self.Task, edge[3], edge[0] + 1)
                raise IOError(msg)
        for edge in recv:
            key = edge[0:2]
            if key in self.Base.send:
                if self.Base.send[key].count > 0:
                    value = self.Base.send[key].value
                    # Todo: add an option to deepcopy(value)
                    if str(type(value)) in self.inputs[edge[3]].types or \
                               len(self.inputs[edge[3]].types)==0:
                        self.inputs[edge[3]].value = value
                        self.inputs[edge[3]].fro = self.Base.send[key].fro
                        self.Base.send[key].count -= 1
                    else:
                        msg = "@Task #%i(%s): The input token '%s' doesn't support the received format"  % (
                            self.iblock + 1, self.Task, edge[3])
                        raise IOError(msg)
                if self.Base.send[key].count == 0:
                    del self.Base.send[key]
            else:
                msg = "@Task #%i(%s): no output has been sent to the input token '%s'" % (
                    self.iblock + 1, self.Task, edge[3])
                raise IOError(msg)

    def Send(self):
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token in self.outputs:
                if self.outputs[token].value is not None:
                    self.outputs[token].count = order.count(token)
                    self.Base.send[(self.iblock, token)] = self.outputs[token]
                else:
                    msg = "@Task #%i(%s): not allowed to send out empty objects '%s'" % (
                        self.iblock + 1, self.Task, token)
                    warnings.warn(msg)
            else:
                msg = "@Task #%i(%s): not a valid output token '%s'" % (self.iblock + 1, self.Task, token)
                raise NameError(msg)

    def required(self, token, req=False):
        """
        Tasks:
            - check if input token is required

        :param token: string, name of the input
        :param req: Boolean, optional (default = False)

        """
        if self.inputs[token].value is None:
            if req:
                msg = "@Task #%i(%s): The input '%s' is required." \
                      % (self.iblock + 1, self.Task, token)
                raise IOError(msg)

    def paramFROMinput(self):
        for param in self.parameters:
            if isinstance(self.parameters[param], str):
                if self.parameters[param][0]=='@':
                    token = self.parameters[param][1:].strip()
                    if token in self.inputs:
                        self.parameters[param] = self.inputs[token].value
                    else:
                        msg = "@Task #%i(%s): assigned an unknown token name - %s - to the parameter - %s - " \
                              % (self.iblock + 1, self.Task, token, param)
                        raise IOError(msg)

    def set_value(self,token,value):
        self.outputs[token].fro = (self.iblock,self.Host,self.Function)
        if str(type(value)) in self.outputs[token].types or \
                    len(self.outputs[token].types)==0:
            self.outputs[token].value = value
        else:
            msg = "@Task #%i(%s): The output token '%s' doesn't support the type"  % (
                self.iblock, self.Host, token)
            raise IOError(msg)

    def import_sklearn(self):
        try:
            exec ("from %s.%s import %s" % (self.metadata.modules[0], self.metadata.modules[1], self.Function))
            submodule = getattr(__import__(self.metadata.modules[0]), self.metadata.modules[1])
            F = getattr(submodule, self.Function)
            api = F(**self.parameters)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(err).__name__ + ': ' + err.message
            raise TypeError(msg)
        return api

    def Fit_sklearn(self):
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

        # methods: fit, predict, None for regression
        # methods: fit_transform, transform, inverse_transform, None for transformers
        if self.method == None:
            api = self.import_sklearn()
            self.set_value('api', api)
        elif self.method == 'fit_transform':
            api = self.import_sklearn()
            self.required('df', req=True)
            df = self.inputs['df'].value
            df = api.fit_transform(df)
            self.set_value('api', api)
            self.set_value('df', pd.DataFrame(df))
        elif self.method == 'transform':
            self.required('df', req=True)
            self.required('api', req=True)
            df = self.inputs['df'].value
            api = self.inputs['api'].value
            df = api.transform(df)
            self.set_value('api', api)
            self.set_value('df', pd.DataFrame(df))
        elif self.method == 'inverse_transform':
            self.required('df', req=True)
            self.required('api', req=True)
            df = self.inputs['df'].value
            api = self.inputs['api'].value
            df = api.inverse_transform(df)
            self.set_value('api', api)
            self.set_value('df', pd.DataFrame(df))
        elif self.method == 'fit':
            api = self.import_sklearn()
            self.required('dfx', req=True)
            self.required('dfy', req=True)
            dfx = self.inputs['dfx'].value
            dfy = self.inputs['dfy'].value
            api.fit(dfx,dfy)
            dfy_predict = api.predict(dfx)
            self.set_value('api', api)
            self.set_value('dfy_predict', pd.DataFrame(dfy_predict))
        elif self.method == 'predict':
            self.required('dfx', req=True)
            self.required('api', req=True)
            dfx = self.inputs['dfx'].value
            api = self.inputs['api'].value
            dfy_predict = api.predict(dfx)
            self.set_value('api', api)
            self.set_value('dfy_predict', pd.DataFrame(dfy_predict))

    def _dim_check(self, token, X, ndim):
        if (X.ndim == ndim < 3):
            pass
        elif (X.ndim == 1) and (ndim == 2):
            X = X.reshape(-1,1)
        elif (X.ndim == 2) and (X.shape[1] == 1) and (ndim == 1):
            X = X.ravel()
        else:
            msg = "@Task #%i(%s): the %s is not or can not be converted to %i dimensional " \
                  % (self.iblock + 1, self.Task, token, ndim)
            raise IOError(msg)
        return X

    def data_check(self, token, X, ndim=2, n0=None, n1=None, format_out='df'):
        """
        Tasks:
            - check the dimension and size of input
            - change the format from numpy array to pandas data frame or vice versa

        :param X: numpy.ndarray or pandas.DataFrame
            input data
        :param token: string
            name of input (e.g. training input)
        :param ndim: integer, optional (default=2)
            X.ndim; valid digits are 1 and 2
        :param n0: int
            number of data entries
        :param n1: int
            number of features
        :param format_out: string ('df' or 'ar'), optional (default = 'df')

        :return input data converted to array or dataframe
        :return the header of dataframe
            if input data is not a dataframe return None
        """
        if isinstance(X, pd.DataFrame):
            if format_out == 'ar':
                # print '%s.ndim:'%token, X.values.ndim, "; changing to %i-dimension ..." %ndim
                header = X.columns
                X = self._dim_check(token, X.values, ndim)
            else:
                header = X.columns
            # if not np.can_cast(X.dtypes, np.float, casting='same_kind'):
            #     msg = "@Task #%i(%s): %s cannot be cast to floats" \
            #           % (self.iblock + 1, self.Task, token)
            #     raise Exception(msg)
        elif isinstance(X, np.ndarray):
            if format_out == 'df':
                X = pd.DataFrame(X)
                header = None
            else:
                header = None
                X = self._dim_check(token, X, ndim)
        else:
            msg = "@Task #%i(%s): %s needs to be either pandas dataframe or numpy array" \
                  % (self.iblock + 1, self.Task, token)
            raise Exception(msg)

        if n0 and X.shape[0] != n0:
            msg = "@Task #%i(%s): %s has an inconsistant number of data entries" \
                  % (self.iblock + 1, self.Task, token)
            raise Exception(msg)
        if n1 and X.shape[1] != n1:
            msg = "@Task #%i(%s): %s has an inconsistant number of feature entries" \
                  % (self.iblock + 1, self.Task, token)
            raise Exception(msg)
        return X, header #X.astype(float), header

class LIBRARY(object):
    """
    Do not instantiate this class
    """
    def references(self,host,function):
        if host == 'sklearn':
            ref_g = "https://github.com/scikit-learn/scikit-learn"
            ref_p = "Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011."
            self.refs['scikit-learn'] = {'github':ref_g, 'paper':ref_p}
        elif host == 'cheml':
            ref_g = "https://mhlari@bitbucket.org/hachmanngroup/cheml.git"
            ref_p = "no publicatin"
            self.refs['ChemML'] =  {'github': ref_g, 'paper': ref_p}
            if function == 'RDKitFingerprint':
                ref_g = "https://github.com/rdkit"
                ref_p = "no publication"
                self.refs['rdkit'] = {'github': ref_g, 'paper': ref_p}
            elif function == 'Dragon':
                ref_g = "http://www.talete.mi.it/products/dragon_description.htm"
                ref_p = "R. Todeschini,  V. Consonni,  R. Mannhold,H.  Kubinyi,  H.  Timmerman,  Handbook  ofMolecular Descriptors, Wiley-VCH, 2000."
                self.refs['Dragon'] = {'url': ref_g, 'paper': ref_p}
            elif function == 'CoulombMatrix':
                ref_g = "no software package"
                ref_p = "M.  Rupp,  A.  Tkatchenko,  K.-R.  Muller,O.  A.  von  Lilienfeld,   Fast  and  Accu-rate  Modeling  of  Molecular  AtomizationEnergies  with  Machine  Learning,   Physi-cal  Review  Letters  108  (5)  (2012)  058301.doi:10.1103/PhysRevLett.108.058301"
                self.refs['CoulombMatrix'] = {'url': ref_g, 'paper': ref_p}
            elif function == 'BagofBonds':
                ref_g = "no software package"
                ref_p = "Hansen, K.; Biegler, F.; Ramakrishnan, R.; Pronobis, W.; von Lilienfeld, O. A.; Muller, K.-R.; Tkatchenko, A. Machine Learning Predictions of Molecular Properties: Accurate Many-Body Potentials and Nonlocality in Chemical Space J. Phys. Chem. Lett. 2015, 6, 2326 2331, DOI: 10.1021/acs.jpclett.5b00831"
                self.refs['BagofBonds'] = {'url': ref_g, 'paper': ref_p}
        elif host == 'tf':
            ref_g = "https://github.com/tensorflow/tensorflow"
            ref_p = "M. Abadi,  P. Barham,  J. Chen,  Z. Chen,A. Davis, J. Dean, M. Devin, S. Ghemawat,G. Irving, M. Isard, M. Kudlur, J. Levenberg,R. Monga, S. Moore, D. G. Murray, B. Steiner,P. Tucker, V. Vasudevan, P. Warden, M. Wicke,Y. Yu, X. Zheng, Tensorflow:  A system forlarge-scale machine learning, in: 12th USENIXSymposium on Operating Systems Design andImplementation (OSDI 16), USENIX Associa-tion, GA, 2016, pp. 265-283"
            self.refs['tensorflow'] = {'github': ref_g, 'paper': ref_p}

    def _save_references(self):
        with open(self.Base.output_directory+'/citation.txt','w') as file:
            for module in self.refs:
                file.write(module+':\n')
                for source in self.refs[module]:
                    file.write('    '+source+': '+self.refs[module][source]+'\n')
                file.write('\n')

    def manual(self, host_function, token=None, slit1=None):
        _ , _ , CMLWinfo = BANK()
        if token:
            if host_function in CMLWinfo:
                if token in CMLWinfo[host_function]:
                    if slit1[1:] not in CMLWinfo[host_function][token]:
                        msg = "@Task #%i(%s): received an illegal input format - %s - for the token %s. " \
                              "The list of acceptable formats are: %s" \
                              % (self.iblock + 1, self.Task, str(slit1[1:]), token, str(CMLWinfo[host_function][token]))
                        raise IOError(msg)
        else:
            if host_function not in CMLWinfo:
                return False
            else:
                return True

def BANK():
    # 7 tasks
    tasks = ['Enter','Prepare','Model','Search','Mix','Visualize','Store']
    # info = {'task':{'subtask':{'host':{'function':{ 'inputs': {'name':'df', 'required':True, 'value':None, 'allowed':[],
    #                                                          'sender':(), 'py_type':pd.DataFrame
    #                                                         }
    #                                  'wrapper_params':{'param':{'required':False, 'value':None}
    #                                                   }
    #                                  'function_params':{'param': {'required': False, 'value': None}
    #                                                    }
    #                                }
    #                      }
    #                }
    #       }
    # }
    # sklearn_regression_function
    skl_regression_func = ['LinearRegression', 'Ridge', 'KernelRidge', 'Lasso', 'MultiTaskLasso', 'ElasticNet', \
                                'MultiTaskElasticNet', 'Lars', 'LassoLars', 'BayesianRidge', 'ARDRegression', \
                                'LogisticRegression', 'SGDRegressor', 'SVR', 'NuSVR', 'LinearSVR', 'MLPRegressor']

    external_info = {('task', 'subtask'):('host', 'function')}
    external_info = {
            'Enter':{
                        'input_data':{
                                    'pandas':['read_excel', 'read_csv'],
                                     }
                    },
            'Prepare':{
                        'descriptor': {'cheml': ['RDKitFingerprint', 'Dragon', 'CoulombMatrix'],
                                       'sklearn': ['PolynomialFeatures', 'Binarizer','OneHotEncoder']
                                       },
                        'scaler': {
                                    'sklearn': ['StandardScaler','MinMaxScaler','MaxAbsScaler','RobustScaler','Normalizer']
                                  },
                        'feature selector': {
                                                'sklearn': ['PCA','KernelPCA']
                                            },
                        'feature transformer': {
                                                'cheml': ['TBFS']
                                                },
                        'basic operator': {
                                        'cheml':['PyScript','Merge','Split', 'Constant','MissingValues','Trimmer','Uniformer'],
                                        'sklearn': ['Imputer']
                                          },
                        'splitter': {
                                        'sklearn': ['Train_Test_Split','KFold']
                                    },
                      },
            'Model':{
                        'regression':{
                                        'cheml':['NN_PSGD','nn_dsgd'],
                                        'sklearn':[
                                                'OLS','Ridge','KernelRidge','Lasso','MultiTaskLasso','',
                                                'ElasticNet','MultiTaskElasticNet','Lars','LassoLars',
                                                'BayesianRidge', 'ARDRegression', 'LogisticRegression',
                                                'SGDRegressor','SVR','NuSVR','LinearSVR','MLPRegressor',
                                                ]
                                        },
                        'classification': {},
                        'clustering': {},
                    },
            'Search':{
                        'evolutionary': {
                                        'cheml': ['GeneticAlgorithm_binary'],
                                        'deep': []
                                        },
                        'swarm': {
                                    'pyswarm': ['pso']
                                 },
                        'grid':{
                                    'sklearn': ['GridSearchCV',]
                                },
                        'metrics':{
                                        'sklearn':['Evaluate_Regression']
                                   },
                     },
            'Mix':{
                    'A': {
                            'sklearn': ['cross_val_score',]
                          },
                    'B': {}
                  },
            'Visualize':{
                            'matplotlib': [],
                            'seaborn': []
                        },
            'Store':{
                        'output_data':{
                                        'cheml': ['SaveFile'],
                                      }
                    }
            }
    internal_info = {('host','function'):{'inputs':{ 'token': 'Input_instance'},
                                          'outputs':{'token': 'Output_instance'},
                                          'wrapper_param':{'param':{'required':True, 'value':None}
                                                           },
                                          'function_param':{'param':{'required':True, 'value':None}
                                                            },
                                         }
                    }

    pddf = "<class 'pandas.core.frame.DataFrame'>"
    internal_info = {
        ('cheml', 'RDKitFingerprint'): {'inputs':{'molfile': str},'outputs':{}},
        ('cheml', 'Dragon'): {'molfile': [('filepath', 'cheml', 'SaveFile'), ]},
        ('cheml', 'CoulombMatrix'): {'': []},
        ('cheml', 'BagofBonds'): {'': []},
        ('cheml', 'PyScript'): {'df1': pddf, 'df2':pddf },

        ('cheml', 'ReadTable'): {'': []},
        ('cheml', 'Merge'): {'df1': [], 'df2': []},
        ('cheml', 'Split'): {'df': []},
        ('cheml', 'SaveFile'): {'df': []},
        ('cheml', 'StoreFile'): {},  # {'input':[]},

        ('cheml', 'MissingValues'): {'dfx': [], 'dfy': []},
        ('cheml', 'Trimmer'): {'': []},
        ('cheml', 'Uniformer'): {'': []},
        ('cheml', 'Constant'): {'df': []},
        ('cheml', 'TBFS'): {'': []},
        ('cheml', 'NN_PSGD'): {'dfx_train': [], 'dfy_train': [], 'dfx_test': []},
        ('cheml', ''): {'': []},
        ('cheml', ''): {'': []},

        ('sklearn', 'SVR'): {},
        ('sklearn', 'MLPRegressor'): {'dfx': [], 'dfy': []},

        ('sklearn', 'PolynomialFeatures'): {'df':pddf, 'api':"<class 'sklearn.preprocessing.data.PolynomialFeatures'>"},
        ('sklearn', 'Binarizer'): {'df':pddf, 'api':"<class 'sklearn.preprocessing.data.Binarizer'>"},
        ('sklearn', 'OneHotEncoder'): {'df': pddf, 'api': "<class 'sklearn.preprocessing.data.OneHotEncoder'>"},
        ('sklearn', 'Imputer'): {'df': pddf, 'api': "<class 'sklearn.preprocessing.imputation.Imputer'>"},
        ('sklearn', 'StandardScaler'): {'df': pddf, 'api':""},
        ('sklearn', 'MinMaxScaler'): {'df': pddf, 'api':""},
        ('sklearn', 'MaxAbsScaler'): {'df': pddf, 'api': ""},
        ('sklearn', 'RobustScaler'): {'df': pddf, 'api': ""},
        ('sklearn', 'RobustScaler'): {'df': pddf, 'api': ""},
        ('sklearn', 'RobustScaler'): {'df': pddf, 'api': ""},
        ('sklearn', 'RobustScaler'): {'df': pddf, 'api': ""},
        ('sklearn', 'Train_Test_Split'): {'dfx': [], 'dfy': []},
        ('sklearn', 'KFold'): {},

        ('sklearn', 'Evaluate_Regression'): {'dfy': [], 'dfy_pred': []},
        ('sklearn', 'scorer_regression'): {},
        ('sklearn', 'ShuffleSplit'): {},
        ('sklearn', 'StratifiedShuffleSplit'): {},
        ('sklearn', 'GridSearchCV'): {},
        ('sklearn', 'learning_curve'): {'dfx': [], 'dfy': []},


        ('pandas', 'read_table'): {'': []},
        ('pandas', 'corr'): {'df': []}

    }

    return tasks, external_info, internal_info