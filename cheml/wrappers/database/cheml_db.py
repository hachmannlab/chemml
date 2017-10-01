import numpy as np
from .containers import Input, Output, Parameter, req, regression_types, cv_types

class Binarizer(object):
    task = 'Prepare'
    subtask = 'python script'
    host = 'cheml'
    function = 'PyScript'
    modules = ('cheml','')
    requirements = (req(0), req(2))
    documentation = ""

    class Inputs:
        iv1 = Input("iv1","input variable, of any format", ())
        iv2 = Input("iv2","input variable, of any format", ())
        iv3 = Input("iv3","input variable, of any format", ())
        iv4 = Input("iv4","input variable, of any format", ())
        iv5 = Input("iv5","input variable, of any format", ())
        iv6 = Input("iv6", "input variable, of any format", ())
    class Outputs:
        ov1 = Output("ov1","output variable, of any format", ())
        ov2 = Output("ov2","output variable, of any format", ())
        ov3 = Output("ov3","output variable, of any format", ())
        ov4 = Output("ov4","output variable, of any format", ())
        ov5 = Output("ov5","output variable, of any format", ())
        ov6 = Output("ov6", "output variable, of any format", ())
    class WParameters:
        func_method = Parameter('func_method','None','string',
                        description = "fit_transform: always make a new api; transform: must receive an api; None: only make a new api ",
                        options = ('fit_transform', 'transform', None))
        track_header = Parameter('track_header', True,'Boolean',
                        description = "if True, the input dataframe's header will be transformed to the output dataframe",
                        options = (True, False))
    class FParameters:
        pass



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