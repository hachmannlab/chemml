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
from cheml import initialization

output_directory, log_file, error_file = initialization.output(output_directory = 'CheML.out',
                                                               logfile = 'log.txt',
                                                               errorfile = 'error.txt')
###########################

########################### MISSING_VALUES
from cheml import preprocessing

missval = preprocessing.missing_values(strategy = 'mean',
                                       string_as_null = True,
                                       inf_as_null = True,
                                       missing_values = False)
data = missval.fit(data)
target = missval.fit(target)
###########################

########################### StandardScaler
from sklearn.preprocessing import StandardScaler

StandardScaler_API = StandardScaler(copy = True,
                                    with_mean = True,
                                    with_std = True)
StandardScaler_API_data, data = preprocessing.transformer_dataframe(transformer = StandardScaler_API,
                                                                    df = data)
###########################

########################### MinMaxScaler
from sklearn.preprocessing import MinMaxScaler

MinMaxScaler_API = MinMaxScaler(feature_range = (0,1),
                                copy = True)
MinMaxScaler_API_data, data = preprocessing.transformer_dataframe(transformer = MinMaxScaler_API,
                                                                  df = data)
###########################

########################### MaxAbsScaler
from sklearn.preprocessing import MaxAbsScaler

MaxAbsScaler_API = MaxAbsScaler(copy = True)
MaxAbsScaler_API_data, data = preprocessing.transformer_dataframe(transformer = MaxAbsScaler_API,
                                                                  df = data)
###########################

########################### RobustScaler
from sklearn.preprocessing import RobustScaler

RobustScaler_API = RobustScaler(with_centering = True,
                                with_scaling = True,
                                copy = True)
RobustScaler_API_data, data = preprocessing.transformer_dataframe(transformer = RobustScaler_API,
                                                                  df = data)
###########################

########################### Normalizer
from sklearn.preprocessing import Normalizer

Normalizer_API = Normalizer(norm = 'l2',
                            copy = True)
Normalizer_API_data, data = preprocessing.transformer_dataframe(transformer = Normalizer_API,
                                                                df = data)
###########################

########################### Binarizer
from sklearn.preprocessing import Binarizer

Binarizer_API = Binarizer(threshold = 0.0,
                          copy = True)
Binarizer_API_data, data = preprocessing.transformer_dataframe(transformer = Binarizer_API,
                                                               df = data)
###########################

########################### OneHotEncoder
from sklearn.preprocessing import OneHotEncoder

OneHotEncoder_API = OneHotEncoder(n_values = 'auto',
                                  categorical_features = 'all',
                                  dtype = np.float,
                                  sparse = True,
                                  handle_unknown = 'error')
OneHotEncoder_API_data, data = preprocessing.transformer_dataframe(transformer = OneHotEncoder_API,
                                                                   df = data)
###########################

########################### PolynomialFeatures
from sklearn.preprocessing import PolynomialFeatures

PolynomialFeatures_API = PolynomialFeatures(degree = 2,
                                            interaction_only = False,
                                            include_bias = True)
PolynomialFeatures_API_data, data = preprocessing.transformer_dataframe(transformer = PolynomialFeatures_API,
                                                                        df = data)
###########################

########################### FunctionTransformer
from sklearn.preprocessing import FunctionTransformer

FunctionTransformer_API = FunctionTransformer(func = None,
                                              validate = True,
                                              accept_sparse = False,
                                              pass_y = False)
FunctionTransformer_API_data, data = preprocessing.transformer_dataframe(transformer = FunctionTransformer_API,
                                                                         df = data)
###########################

########################### VarianceThreshold
from sklearn.feature_selection import VarianceThreshold

VarianceThreshold_API = VarianceThreshold(threshold = 0.0)
VarianceThreshold_API_data, data = preprocessing.VarianceThreshold_dataframe(transformer = VarianceThreshold_API,
                                                                             df = data)
###########################

