class Input(object):
    """Container class for inputs of a function"""
    def __init__(self,name, short_description, types):
        self.name = name
        self.short_description = short_description
        self.long_description = ''
        self.value = None
        self.types = types

class Output(object):
    """Container class for outputs of a function"""
    def __init__(self,name, short_description, types):
        self.name = name
        self.short_description = short_description
        self.long_description = ''
        self.value = None
        self.types = types
        self.fro = ('iblock', 'host', 'function')
        self.count = 0


class Parameter(object):
    """Container class for parameters of a function"""
    def __init__(self,name,default,format='',required=False,description="",options=[]):
        self.name = name
        self.default = default
        self.format = format
        self.required = required
        self.description = description
        self.options = options

def req(ind):
    all_req = { 0 : ('ChemML','0.8'),
                1 : ('scikit-learn','0.19.0'),
                2 : ('pandas', '1.1.3'),
                3 : ('RDKit','2016.03.1'),
                4 : ('Dragon','7 or 6'),
                5 : ('lxml', '3.4.0'),
                6 : ('Babel', '2.3.4'),
                7 : ('matplotlib','1.5.1'),
                8 : ('keras', '2.1.2'),
                9 : ('tensorflow', '1.4.1'),
                10: ('geneticalgorithm', '2.0')
              }
    return all_req[ind]

#conda install -c conda-forge -c rdkit numpy pandas scikit-learn tensorflow keras rdkit babel deap matplotlib lxml ipywidgets graphviz


def regression_types():
    """all the regression classes that follow the sklearn format"""
    sklearn_types = ["<class 'sklearn.linear_model.base.LinearRegression'>","<class 'sklearn.linear_model.ridge.Ridge'>",
                 "<class 'sklearn.kernel_ridge.KernelRidge'>", "<class 'sklearn.linear_model.coordinate_descent.Lasso'>",
                 "<class 'sklearn.linear_model.coordinate_descent.MultiTaskLasso'>", "<class 'sklearn.linear_model.coordinate_descent.ElasticNet'>",
                 "<class 'sklearn.linear_model.coordinate_descent.MultiTaskElasticNet'>", "<class 'sklearn.linear_model.least_angle.Lars'>",
                 "<class 'sklearn.linear_model.least_angle.LassoLars'>", "<class 'sklearn.linear_model.bayes.BayesianRidge'>",
                 "<class 'sklearn.linear_model.bayes.ARDRegression'>", "<class 'sklearn.linear_model.logistic.LogisticRegression'>",
                 "<class 'sklearn.linear_model.stochastic_gradient.SGDRegressor'>", "<class 'sklearn.svm.classes.SVR'>",
                 "<class 'sklearn.svm.classes.NuSVR'>", "<class 'sklearn.svm.classes.LinearSVR'>", "<class 'sklearn.neural_network._multilayer_perceptron.MLPRegressor'>"]
    chemml_types = ["<class 'chemml.nn.keras.mlp.MLP_sklearn'>"]
    all_types = tuple(sklearn_types+chemml_types)
    return all_types

def cv_classes():
    """all objects to be used as a cross-validation generator"""
    all_types = ("<class 'sklearn.model_selection._split.KFold'>", "<class 'sklearn.model_selection._split.ShuffleSplit'>",
                    "<class 'sklearn.model_selection._split.StratifiedShuffleSplit'>",
                 "<class 'sklearn.model_selection._split.LeaveOneOut'>")

    return all_types

###########

def merge_dicts(*dict_args):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result