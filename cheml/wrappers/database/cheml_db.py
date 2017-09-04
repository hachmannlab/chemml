import numpy as np
import pandas as pd

class Input(object):
    """Container class for inputs of a function"""
    def __init__(self,name, info_short):
        self.name = name
        self.info_short = info_short
        self.info_long = ''
        self.value = None
        self.type = None
        self.sender = None
    def set(self):
        pass

class Output(object):
    """Container class for outputs of a function"""
    def __init__(self,name, info_short):
        self.name = name
        self.info_short = info_short
        self.info_long = ''
        self.value = None
        self.type = None
        self.count = 0
        self.sender = None

class HF(object):
    """Container class for a function"""
    def __init__(self):
        self.task = None
        self.subtask = None
        self.host = None
        self.function = None
        self.inputs = {}
        self.outputs = {}
        self.wrapper_parameters = {}
        self.function_parameters = {}
        self.parameters_doc = None
        self.requirements = []


def cml_db():
    internal_info = {('host','function'):{'inputs':{ 'token': Input},
                                          'outputs':{'token': Output},
                                          'wrapper_param':{'param':{'required':True, 'value':None}
                                                           },
                                          'function_param':{'param':{'required':True, 'value':None}
                                                            },
                                          'reqruirements':[]
                                         }
                    }

    #('sklearn', 'PolynomialFeatures')
    if True:
        inp1 = Input('df', 'pandas dataframe')
        inp1.type = ["<class 'pandas.core.frame.DataFrame'>"]
        inp2 = Input('api', 'sklearn PolynomialFeatures class')
        inp2.type = ["<class 'sklearn.preprocessing.data.PolynomialFeatures'>"]
        hf = HF()
        hf.inputs = {'df':inp1,'api':inp2}
        hf.outputs = {'df':Output('df'),'api':Output('api')}
        hf.wrapper_parameters = {'func_method':('fit','fit_transform','transform',None)}
        hf.requirements+=['sklearn']
        internal_info[('sklearn', 'PolynomialFeatures')] = hf

    internal_info = {
        ('cheml', 'RDKitFingerprint'): {'inputs':{'molfile': str},'outputs':{}},
        ('cheml', 'Dragon'): {'molfile': [('filepath', 'cheml', 'SaveFile'), ]},
        ('cheml', 'CoulombMatrix'): {'': []},
        ('cheml', 'BagofBonds'): {'': []},
        ('cheml', 'PyScript'): {'df1': pd.DataFrame},

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

        ('sklearn', 'Evaluate_Regression'): {'dfy': [], 'dfy_pred': []},
        ('sklearn', 'scorer_regression'): {},
        ('sklearn', 'Train_Test_Split'): {'dfx': [], 'dfy': []},
        ('sklearn', 'ShuffleSplit'): {},
        ('sklearn', 'StratifiedShuffleSplit'): {},
        ('sklearn', 'GridSearchCV'): {},
        ('sklearn', 'learning_curve'): {'dfx': [], 'dfy': []},

        ('sklearn', 'StandardScaler'): {'df': []},
        ('sklearn', 'Binarizer'): {'df': []},

        ('pandas', 'read_table'): {'': []},
        ('pandas', 'corr'): {'df': []}

    }

    return internal_info