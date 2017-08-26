import numpy as np
import pandas as pd

## Note: Pandas
class BASE(object):
    """
    Do not instantiate this class
    """
    def __init__(self, Base, parameters, iblock, SuperFunction, Function, Host):
        self.Base = Base
        self.parameters = parameters
        self.iblock = iblock
        self.SuperFunction = SuperFunction
        self.Function = Function
        self.Host = Host

    def run(self):
        self.legal_IO()
        self.receive()
        self.fit()

    def receive(self):
        recv = [edge for edge in self.Base.graph if edge[2] == self.iblock]
        self.Base.graph = tuple([edge for edge in self.Base.graph if edge[2] != self.iblock])
        # check received tokens to: (1) be a legal input, and (2) be unique.
        count = {token: 0 for token in self.legal_inputs}
        for edge in recv:
            if edge[3] in self.legal_inputs:
                count[edge[3]] += 1
                if count[edge[3]] > 1:
                    msg = '@Task #%i(%s): only one input per each available input path/token can be received.' % (
                        self.iblock + 1, self.SuperFunction)
                    raise IOError(msg)
            else:
                msg = "@Task #%i(%s): received a non valid input token '%s', sent by function #%i" % (
                    self.iblock + 1, self.SuperFunction, edge[3], edge[0] + 1)
                raise IOError(msg)
        for edge in recv:
            key = edge[0:2]
            if key in self.Base.send:
                if self.Base.send[key][1] > 0:
                    value = self.Base.send[key][0]
                    # value = copy.deepcopy(self.Base.send[key][0])
                    # else:
                    #     value = self.Base.send[key][0]
                    # Todo: informative token should be a list of (int(edge[0]),edge[1])
                    # informative_token = (int(edge[0]), edge[1]) + self.Base.graph_info[int(edge[0])]
                    self.legal_inputs[edge[3]] = (value, self.Base.send[key][2])
                    del value
                    self.Base.send[key][1] -= 1
                if self.Base.send[key][1] == 0:
                    del self.Base.send[key]
            else:
                msg = '@Task #%i(%s): broken pipe in token %s - nothing has been sent' % (
                    self.iblock + 1, self.SuperFunction, edge[3])
                raise IOError(msg)
        return self.legal_inputs

    def _error_type(self, token):
        msg = "@Task #%i(%s): The type of input '%s' is not valid" \
              % (self.iblock + 1, self.SuperFunction, token)
        raise IOError(msg)

    def input_check(self, token, req=False, py_type=False):
        """
        Tasks:
            - check if input token is required
            - check if python type is correct
            - check if the input is acceptable (based on the original sender)

        Note:
            - always run with Library.manual

        :param token: string, name of the input
        :param req: Boolean, optional (default = False)
        :param py_type: any python recognizable type, optional (default = False)
        :return:
            token value and token information (the sender info)
        """
        if self.legal_inputs[token] is None:
            if req:
                msg = "@Task #%i(%s): The input '%s' is required." \
                      % (self.iblock + 1, self.SuperFunction, token)
                raise IOError(msg)
            else:
                return None, None
        else:
            slit0 = self.legal_inputs[token][0]
            slit1 = self.legal_inputs[token][1]
            if py_type:
                if not isinstance(slit0, py_type):
                    self._error_type(token)
            else:
                self.manual(host_function = self.Base.graph_info[self.iblock], token=token, slit1=slit1)
        return slit0, slit1

    def paramFROMinput(self):
        for param in self.parameters:
            if isinstance(self.parameters[param], str):
                if self.parameters[param][0]=='@':
                    token = self.parameters[param][1:].strip()
                    if token in self.legal_inputs:
                        self.parameters[param] = self.legal_inputs[token][0]
                    else:
                        msg = "@Task #%i(%s): assigned an unknown token name - %s - to the parameter - %s - " \
                              % (self.iblock + 1, self.SuperFunction, token, param)
                        raise IOError(msg)

    def _dim_check(self, token, X, ndim):
        if (X.ndim == ndim < 3):
            pass
        elif (X.ndim == 1) and (ndim == 2):
            X = np.array([[i] for i in X])
        elif (X.ndim == 2) and (X.shape[1] == 1) and (ndim == 1):
            X = X.ravel()
        else:
            msg = "@Task #%i(%s): the %s is not or can not be converted to %i dimensional " \
                  % (self.iblock + 1, self.SuperFunction, token, ndim)
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
                print '%s.ndim:'%token, X.values.ndim, "; changing to %i-dimension ..." %ndim
                header = X.columns
                X = self._dim_check(token, X.values, ndim)
            else:
                header = X.columns
            # if not np.can_cast(X.dtypes, np.float, casting='same_kind'):
            #     msg = "@Task #%i(%s): %s cannot be cast to floats" \
            #           % (self.iblock + 1, self.SuperFunction, token)
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
                  % (self.iblock + 1, self.SuperFunction, token)
            raise Exception(msg)

        if n0 and X.shape[0] != n0:
            msg = "@Task #%i(%s): %s has an invalid number of data entries" \
                  % (self.iblock + 1, self.SuperFunction, token)
            raise Exception(msg)
        if n1 and X.shape[1] != n1:
            msg = "@Task #%i(%s): %s has an invalid number of feature entries" \
                  % (self.iblock + 1, self.SuperFunction, token)
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
        # legal_modules = {'cheml':['RDKitFingerprint','Dragon','CoulombMatrix','BagofBonds',
        #                           'PyScript','File','Merge','Split','SaveFile',
        #                           'MissingValues','Trimmer','Uniformer','Constant','TBFS',
        #                           'NN_PSGD','NN_DSGD','NN_MLP_Theano','NN_MLP_Tensorflow','SVR'],
        #                  'sklearn': ['PolynomialFeatures','Imputer','StandardScaler','MinMaxScaler','MaxAbsScaler',
        #                              'RobustScaler','Normalizer','Binarizer','OneHotEncoder',
        #                              'VarianceThreshold','SelectKBest','SelectPercentile','SelectFpr','SelectFdr',
        #                              'SelectFwe','RFE','RFECV','SelectFromModel',
        #                              'PCA','KernelPCA','RandomizedPCA','LDA',
        #                              'Train_Test_Split','KFold','','','',
        #                              'SVR','','',
        #                              'GridSearchCV','Evaluation',''],
        #                  'tf':[]}


        # sklearn_regression_function
        self.skl_regression_func = ['OLS','Ridge','KernelRidge','Lasso','MultiTaskLasso','ElasticNet', \
                                    'MultiTaskElasticNet', 'Lars', 'LassoLars','BayesianRidge','ARDRegression', \
                                    'LogisticRegression', 'SGDRegressor', 'SVR','NuSVR','LinearSVR','MLPRegressor']

        # general output formats

        # {inputs: legal output formats}
        CMLWinfo = {
            ('cheml','RDKitFingerprint'):{'molfile':[('filepath', 'cheml', 'SaveFile'),]},
            ('cheml','Dragon'):{'molfile':[('filepath', 'cheml', 'SaveFile'),]},
            ('cheml','CoulombMatrix'):{'':[]},
            ('cheml','BagofBonds'):{'':[]},
            ('cheml','PyScript'):{'':[]},

            ('cheml','ReadTable'):{'':[]},
            ('cheml','Merge'):{'df1':[], 'df2':[]},
            ('cheml','Split'):{'df':[]},
            ('cheml','SaveFile'):{'df':[]},
            ('cheml', 'StoreFile'): {},#{'input':[]},

            ('cheml','MissingValues'):{'dfx':[], 'dfy':[]},
            ('cheml','Trimmer'):{'':[]},
            ('cheml','Uniformer'):{'':[]},
            ('cheml','Constant'):{'df':[]},
            ('cheml','TBFS'):{'':[]},
            ('cheml','NN_PSGD'):{'dfx_train':[], 'dfy_train':[], 'dfx_test':[]},
            ('cheml',''):{'':[]},
            ('cheml',''):{'':[]},

            ('sklearn', 'SVR'): {},
            ('sklearn', 'MLPRegressor'): {'dfx':[], 'dfy':[]},

            ('sklearn', 'Evaluate_Regression'): {'dfy':[], 'dfy_pred':[]},
            ('sklearn', 'scorer_regression'):{},
            ('sklearn', 'Train_Test_Split'): {'dfx':[], 'dfy':[]},
            ('sklearn', 'ShuffleSplit'): {},
            ('sklearn', 'StratifiedShuffleSplit'): {},
            ('sklearn', 'GridSearchCV'): {},
            ('sklearn', 'learning_curve'): {'dfx':[], 'dfy':[]},

            ('sklearn', 'StandardScaler'): {'df':[]},
            ('sklearn', 'Binarizer'): {'df':[]},

            ('pandas', 'read_table'): {'': []},
            ('pandas', 'corr'): {'df': []}

        }
        if token:
            if host_function in CMLWinfo:
                if token in CMLWinfo[host_function]:
                    if slit1[1:] not in CMLWinfo[host_function][token]:
                        msg = "@Task #%i(%s): received an illegal input format - %s - for the token %s. " \
                              "The list of acceptable formats are: %s" \
                              % (self.iblock + 1, self.SuperFunction, str(slit1[1:]), token, str(CMLWinfo[host_function][token]))
                        raise IOError(msg)
        else:
            if host_function not in CMLWinfo:
                return False
            else:
                return True

def BANK():
    tasks = ['Enter','Prepare','Model','Search','Mix','Visualize','Store']
    info = {
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
    return info, tasks