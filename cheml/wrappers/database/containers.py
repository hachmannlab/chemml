class Input(object):
    """Container class for inputs of a function"""
    def __init__(self,name, short_description, types):
        self.name = name
        self.short_description = short_description
        self.long_description = ''
        self.value = None
        self.types = types

    def receive(self, graph):
        print 'received'

class Output(object):
    """Container class for outputs of a function"""
    def __init__(self,name, short_description, types):
        self.name = name
        self.short_description = short_description
        self.long_description = ''
        self.value = None
        self.types = types

    def send(self, graph):
        self.count = 0

class Parameter(object):
    """Container class for parameters of a function"""
    def __init__(self,name,default,format,required=False,options=[]):
        self.name = name
        self.default = default
        self.format = format
        self.required = required
        self.options = options

def req(ind):
    all_req = { 0 : ('ChemML','0.1.0'),
                1 : ('scikit-learn','0.19.0'),
                2 : ('pandas', '0.20.3'),
                3 : (),
              }
    return all_req[ind]

def regression_types():
    """all the regression classes that follow the sklearn format"""
    sklearn_types = ["<class 'sklearn.linear_model.base.LinearRegression'>","<class 'sklearn.linear_model.ridge.Ridge'>",
                 "<class 'sklearn.kernel_ridge.KernelRidge'>", "<class 'sklearn.linear_model.coordinate_descent.Lasso'>",
                 "<class 'sklearn.linear_model.coordinate_descent.MultiTaskLasso'>", "<class 'sklearn.linear_model.coordinate_descent.ElasticNet'>",
                 "<class 'sklearn.linear_model.coordinate_descent.MultiTaskElasticNet'>", "<class 'sklearn.linear_model.least_angle.Lars'>",
                 "<class 'sklearn.linear_model.least_angle.LassoLars'>", "<class 'sklearn.linear_model.bayes.BayesianRidge'>",
                 "<class 'sklearn.linear_model.bayes.ARDRegression'>", "<class 'sklearn.linear_model.logistic.LogisticRegression'>",
                 "<class 'sklearn.linear_model.stochastic_gradient.SGDRegressor'>", "<class 'sklearn.svm.classes.SVR'>",
                 "<class 'sklearn.svm.classes.NuSVR'>", "<class 'sklearn.svm.classes.LinearSVR'>", "<class 'sklearn.neural_network.multilayer_perceptron.MLPRegressor'>"]
    cheml_types = []
    all_types = tuple(sklearn_types+cheml_types)
    return all_types

def cv_types():
    """all objects to be used as a cross-validation generator"""
    all_types = ("<class 'sklearn.model_selection._split.KFold'>", "<class 'sklearn.model_selection._split.ShuffleSplit'>",
                    "<class 'sklearn.model_selection._split.StratifiedShuffleSplit'>")

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

def DB():
    # 7 tasks
    tasks = ['Enter','Prepare','Model','Search','Mix','Visualize','Store']
    external_info = {
            'Enter':{
                        'input_data':{
                                    'pandas':['read_excel', 'read_csv'],
                                     }
                    },
            'Prepare':{
                        'feature representation': {'cheml': ['RDKitFingerprint', 'Dragon', 'CoulombMatrix'],
                                       'sklearn': ['PolynomialFeatures', 'Binarizer','OneHotEncoder']
                                       },
                        'scale': {
                                    'sklearn': ['StandardScaler','MinMaxScaler','MaxAbsScaler','RobustScaler','Normalizer']
                                  },
                        'feature selection': {
                                                'sklearn': ['PCA','KernelPCA']
                                            },
                        'feature transformation': {
                                                'cheml': ['TBFS']
                                                },
                        'basic operator': {
                                        'cheml':['PyScript','Merge','Split', 'Constant','MissingValues','Trimmer','Uniformer'],
                                        'sklearn': ['Imputer']
                                          },
                        'split': {
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
                                    'sklearn': ['GridSearchCV','learning_curve']
                                },
                        'validate': {'sklearn':['cross_val_score','cross_val_predict']
                                     },
                        'evaluate':{
                                        'sklearn':['scorer_regression','evaluate_regression']
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

    return tasks, external_info