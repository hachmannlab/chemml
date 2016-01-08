########################### INPUT
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
from sklearn.preprocessing import Imputer
imp = Imputer(strategy = 'mean',
              missing_values = 'NaN',
              axis = 0,
              verbose = 0,
              copy = True)
imp_data, data = preprocessing.Imputer_dataframe(imputer = imp, df = data)
imp_target, target = preprocessing.Imputer_dataframe(imputer = imp, df = target)
###########################

########################### StandardScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler(copy = True,
                        with_mean = True,
                        with_std = True)
data_scaler, data = preprocessing.Scaler_dataframe(scaler = scaler, df = data)
target_scaler, target = preprocessing.Scaler_dataframe(scaler = scaler, df = target)
###########################

########################### MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler(feature_range = 0,
                              copy = True)
data_min_max_scaler, data = preprocessing.Scaler_dataframe(scaler = min_max_scaler, df = data)
target_min_max_scaler, target = preprocessing.Scaler_dataframe(scaler = min_max_scaler, df = target)
###########################

########################### MaxAbsScaler
from sklearn.preprocessing import MaxAbsScaler
max_abs_scaler = MaxAbsScaler(copy = True)
data_min_max_scaler, data = preprocessing.Scaler_dataframe(scaler = max_abs_scaler, df = data)
target_min_max_scaler, target = preprocessing.Scaler_dataframe(scaler = max_abs_scaler, df = target)
###########################

