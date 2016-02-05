########################### INPUT
import numpy as np
import pandas as pd

data = pd.read_csv('benchmarks/homo_dump/sample_50/data_NOsmi_50.csv',
                   sep = None,
                   skiprows = 0,
                   header = 0)
target = pd.read_csv('benchmarks/homo_dump/sample_50/homo_50.csv',
                     sep = None,
                     skiprows = 0,
                     header = None)
###########################

########################### OUTPUT
from cheml.initialization import output

output_directory, log_file, error_file = output(output_directory = 'CheML.out',
                                                logfile = 'log.txt',
                                                errorfile = 'error.txt')
###########################

########################### MISSING_VALUES
from cheml.preprocessing import missing_values
from sklearn.preprocessing import Imputer
from cheml.preprocessing import Imputer_dataframe

missing_values_API = missing_values(strategy = 'mean',
                                    inf_as_null = True,
                                    string_as_null = True,
                                    missing_values = False)
data = missing_values_API.fit(data)
target = missing_values_API.fit(target)
Imputer_API = Imputer(strategy = 'mean',
                      inf_as_null = True,
                      string_as_null = True,
                      missing_values = False)
data = Imputer_dataframe(transformer = Imputer_API,
                         df = data)
target = Imputer_dataframe(transformer = Imputer_API,
                           df = target)
###########################

########################### StandardScaler
from sklearn.preprocessing import StandardScaler
from cheml.preprocessing import transformer_dataframe

StandardScaler_API = StandardScaler(copy = True,
                                    with_mean = True,
                                    with_std = True)
data = transformer_dataframe(transformer = StandardScaler_API,
                             df = data)
###########################

########################### MinMaxScaler
from sklearn.preprocessing import MinMaxScaler

MinMaxScaler_API = MinMaxScaler(copy = True,
                                feature_range = (0,1))
data = transformer_dataframe(transformer = MinMaxScaler_API,
                             df = data)
###########################

########################### MaxAbsScaler
from sklearn.preprocessing import MaxAbsScaler

MaxAbsScaler_API = MaxAbsScaler(copy = True)
data = transformer_dataframe(transformer = MaxAbsScaler_API,
                             df = data)
###########################

########################### RobustScaler
from sklearn.preprocessing import RobustScaler

RobustScaler_API = RobustScaler(with_centering = True,
                                copy = True,
                                with_scaling = True)
data = transformer_dataframe(transformer = RobustScaler_API,
                             df = data)
###########################

########################### Normalizer
from sklearn.preprocessing import Normalizer

Normalizer_API = Normalizer(copy = True,
                            norm = 'l2')
data = transformer_dataframe(transformer = Normalizer_API,
                             df = data)
###########################

########################### Binarizer
from sklearn.preprocessing import Binarizer

Binarizer_API = Binarizer(threshold = 0.0,
                          copy = True)
data = transformer_dataframe(transformer = Binarizer_API,
                             df = data)
###########################

########################### OneHotEncoder
from sklearn.preprocessing import OneHotEncoder

OneHotEncoder_API = OneHotEncoder(dtype = np.float,
                                  handle_unknown = 'error',
                                  sparse = True,
                                  categorical_features = 'all',
                                  n_values = 'auto')
data = transformer_dataframe(transformer = OneHotEncoder_API,
                             df = data)
###########################

########################### PolynomialFeatures
from sklearn.preprocessing import PolynomialFeatures

PolynomialFeatures_API = PolynomialFeatures(include_bias = True,
                                            interaction_only = False,
                                            degree = 2)
data = transformer_dataframe(transformer = PolynomialFeatures_API,
                             df = data)
###########################

########################### FunctionTransformer
from sklearn.preprocessing import FunctionTransformer

FunctionTransformer_API = FunctionTransformer(validate = True,
                                              accept_sparse = False,
                                              func = None,
                                              pass_y = False)
###########################

########################### VarianceThreshold
from sklearn.feature_selection import VarianceThreshold
from cheml.preprocessing import selector_dataframe

VarianceThreshold_API = VarianceThreshold(threshold = 0.0)
data = selector_dataframe(transformer = VarianceThreshold_API,
                          df = data,
                          tf = target)
###########################

########################### SelectKBest
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

SelectKBest_API = SelectKBest(k = 10,
                              score_func = f_regression)
data = selector_dataframe(transformer = SelectKBest_API,
                          df = data,
                          tf = target)
###########################

########################### SelectPercentile
from sklearn.feature_selection import SelectPercentile

SelectPercentile_API = SelectPercentile(percentile = 10,
                                        score_func = f_regression)
data = selector_dataframe(transformer = SelectPercentile_API,
                          df = data,
                          tf = target)
###########################

########################### SelectFpr
from sklearn.feature_selection import SelectFpr

SelectFpr_API = SelectFpr(alpha = 0.05,
                          score_func = f_regression)
data = selector_dataframe(transformer = SelectFpr_API,
                          df = data,
                          tf = target)
###########################

########################### SelectFdr
from sklearn.feature_selection import SelectFdr

SelectFdr_API = SelectFdr(alpha = 0.05,
                          score_func = f_regression)
data = selector_dataframe(transformer = SelectFdr_API,
                          df = data,
                          tf = target)
###########################

########################### SelectFwe
from sklearn.feature_selection import SelectFwe

SelectFwe_API = SelectFwe(alpha = 0.05,
                          score_func = f_regression)
data = selector_dataframe(transformer = SelectFwe_API,
                          df = data,
                          tf = target)
###########################

########################### RFE
from sklearn.feature_selection import RFE
from sklearn.sth import fake1

fake1_API = fake1(a = 1,
                  c = 'mean',
                  b = True,
                  d = ['r',1])
RFE_API = RFE(step = 1,
              estimator = fake1_API,
              verbose = 0,
              estimator_params = None,
              n_features_to_select = None)
data = selector_dataframe(transformer = RFE_API,
                          df = data,
                          tf = target)
###########################

########################### RFECV
from sklearn.feature_selection import RFECV
from sklearn.sth import fake2

fake2_API = fake2(a = 1,
                  c = 'mean',
                  b = True,
                  d = ['r',1])
RFECV_API = RFECV(scoring = None,
                  verbose = 0,
                  step = 1,
                  estimator_params = None,
                  estimator = fake2_API,
                  cv = None)
data = selector_dataframe(transformer = RFECV_API,
                          df = data,
                          tf = target)
###########################

########################### SelectFromModel
from sklearn.feature_selection import SelectFromModel
from sklearn.sth import fake3

fake3_API = fake3(a = 1,
                  c = 'mean',
                  b = True,
                  d = ['r',1])
fake3_API = fake3_API.fit(data, target)
SelectFromModel_API = SelectFromModel(threshold = None,
                                      estimator = fake3_API,
                                      prefit = True)
data = selector_dataframe(transformer = SelectFromModel_API,
                          df = data,
                          tf = target)
###########################

########################### Trimmer
from cheml.initializtion import Trimmer

Trimmer_API = Trimmer(sort = True,
                      shuffle = True,
                      cut = 0.05,
                      type = "margins")
data = Trimmer_API.fit_transform(data, target)
###########################

########################### Uniformer
from cheml.initializtion import Uniformer

Uniformer_API = Uniformer(include_lowest = True,
                          right = True,
                          bin_pop = 0.5,
                          substitute = None,
                          bins = )
data = Uniformer_API.fit_transform(data, target)
###########################

########################### PCA
from sklearn.decomposition import PCA

PCA_API = PCA(copy = True,
              n_components = None,
              whiten = False)
data = PCA_API.fit_transform(data)
data = pd.DataFrame(data)
###########################

########################### KernelPCA
from sklearn.decomposition import KernelPCA

KernelPCA_API = KernelPCA(fit_inverse_transform = False,
                          kernel = "linear",
                          tol = 0,
                          degree = 3,
                          max_iter = None,
                          kernel_params = None,
                          remove_zero_eig = True,
                          n_components = None,
                          eigen_solver = 'auto',
                          alpha = 1.0,
                          coef0 = 1,
                          gamma = None)
data = KernelPCA_API.fit_transform(data)
data = pd.DataFrame(data)
###########################

