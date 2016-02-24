########################### Dragon
from cheml.chem import dragon

dragon_API = dragon(SaveStdOut = False,
                    MaxSR = "35",
                    SaveFilePath = "Dragon_descriptors.txt",
                    SaveExcludeMisMolecules = False,
                    SaveExcludeStdDev = False,
                    SaveExcludeNearConst = False,
                    SaveProject = False,
                    Add2DHydrogens = False,
                    blocks = range(1,30),
                    SaveProjectFile = "Dragon_project.drp",
                    SaveOnlyData = False,
                    SaveExclusionOptionsToVariables = False,
                    LogEdge = True,
                    LogPathWalk = True,
                    SaveLabelsOnSeparateFile = False,
                    version = 6,
                    DefaultMolFormat = "1",
                    molFile = None,
                    HelpBrowser = "/usr/bin/xdg-open",
                    SaveExcludeRejectedMolecules = False,
                    RejectUnusualValence = False,
                    SaveStdDevThreshold = "0.0001",
                    SaveExcludeConst = False,
                    SaveFormatSubBlock = "%b-%s-%n-%m.txt",
                    Decimal_Separator = ".",
                    SaveExcludeCorrelated = False,
                    MaxSRDetour = "30",
                    consecutiveDelimiter = False,
                    molInputFormat = "SMILES",
                    SaveExcludeAllMisVal = False,
                    Weights = ["Mass","VdWVolume","Electronegativity","Polarizability","Ionization","I-State"],
                    external = False,
                    MaxSRforAllCircuit = "19",
                    fileName = None,
                    Missing_String = "Nan",
                    SaveExcludeMisVal = False,
                    logFile = "Dragon_log.txt",
                    SaveLayout = True,
                    molInput = "stdin",
                    SaveFormatBlock = "%b-%n.txt",
                    MissingValue = "NaN",
                    SaveType = "singlefile",
                    ShowWorksheet = False,
                    delimiter = ",",
                    CheckUpdates = True,
                    MaxAtomWalkPath = "2000",
                    logMode = "file",
                    SaveCorrThreshold = "0.95",
                    SaveFile = True)
dragon_API.script_wizard(script = "Dragon_script.drs")
dragon_API.run()
data_path = dragon_API.drs
###########################

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

########################### RandomizedPCA
from sklearn.decomposition import RandomizedPCA

RandomizedPCA_API = RandomizedPCA(random_state = None,
                                  copy = True,
                                  n_components = None,
                                  iterated_power = 3,
                                  whiten = False)
data = RandomizedPCA_API.fit_transform(data)
data = pd.DataFrame(data)
###########################

########################### LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

LinearDiscriminantAnalysis_API = LinearDiscriminantAnalysis(solver = 'svd',
                                                            shrinkage = None,
                                                            n_components = None,
                                                            tol = 0.0001,
                                                            priors = None,
                                                            store_covariance = False)
data = LinearDiscriminantAnalysis_API.fit_transform(data)
data = pd.DataFrame(data)
###########################

########################### SupervisedLearning_regression
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import K-fold
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.svm import NuSVR
from sklearn.svm import LinearSVR


# split
data_train, data_test, target_train, target_test = train_test_split(data,
                                                                    target,
                                                                    train_size = None,
                                                                    random_state = None,
                                                                    test_size = None,
                                                                    stratify = None)

# cross_validation
CV_indices = K-fold(shuffle = False,
                    n = len(data),
                    random_state = None,
                    n_folds = 3)

# scaler
StandardScaler_API = StandardScaler(copy = True,
                                    with_std = True,
                                    with_mean = True)

# learner
SVR_API = SVR(kernel = 'rbf',
              C = 1.0,
              verbose = False,
              shrinking = True,
              epsilon = 0.1,
              max_iter = -1,
              tol = 1e-3,
              cache_size = 200,
              degree s = 3,
              coef0 = 0.0,
              gamma = 'auto')

# result: split
StandardScaler_API.fit(data_train)
data_train = StandardScaler_API.transform(data_train)
data_test = StandardScaler_API.transform(data_test)
SVR_API.fit(data_train, target_train)
target_train_pred = SVR_API.predict(data_train)
target_test_pred = SVR_API.predict(data_test)
split_metrics = {'training':{}, 'test':{}}
split_metrics['training']['r2_score'] = r2_score(target_train, target_train_pred)
split_metrics['test']['r2_score'] = r2_score(target_test, target_test_pred)
split_metrics['training']['mean_absolute_error'] = mean_absolute_error(target_train, target_train_pred)
split_metrics['test']['mean_absolute_error'] = mean_absolute_error(target_test, target_test_pred)
split_metrics['training']['median_absolute_error'] = median_absolute_error(target_train, target_train_pred)
split_metrics['test']['median_absolute_error'] = median_absolute_error(target_test, target_test_pred)
split_metrics['training']['mean_squared_error'] = mean_squared_error(target_train, target_train_pred)
split_metrics['test']['mean_squared_error'] = mean_squared_error(target_test, target_test_pred)
split_metrics['training']['mean_squared_error'] = np.sqrt(mean_squared_error(target_train, target_train_pred))
split_metrics['test']['mean_squared_error'] = np.sqrt(mean_squared_error(target_test, target_test_pred))
split_metrics['training']['explained_variance_score'] = explained_variance_score(target_train, target_train_pred)
split_metrics['test']['explained_variance_score'] = explained_variance_score(target_test, target_test_pred)

# result: cross_validation
CV_metrics = {'test': {'r2_score': [], 'mean_absolute_error': [], 'median_absolute_error': [], 'mean_squared_error': [], 'root_mean_squared_error': [], 'explained_variance_score': []}, 'training': {'r2_score': [], 'mean_absolute_error': [], 'median_absolute_error': [], 'mean_squared_error': [], 'root_mean_squared_error': [], 'explained_variance_score': []}}
for train_index, test_index in CV_indices:
    data_train = data.iloc[train_index,:]
    target_train = target.iloc[train_index,:]
    data_test = target.iloc[test_index,:]
    target_test = target.iloc[test_index,:]
    StandardScaler_API.fit(data_train)
    data_train = StandardScaler_API.transform(data_train)
    data_test = StandardScaler_API.transform(data_test)
    SVR_API.fit(data_train, target_train)
    target_train_pred = SVR_API.predict(data_train)
    target_test_pred = SVR_API.predict(data_test)
    CV_metrics['training']['r2_score'].append(r2_score(target_train, target_train_pred))
    CV_metrics['test']['r2_score'].append(r2_score(target_test, target_test_pred))
    CV_metrics['training']['mean_absolute_error'].append(mean_absolute_error(target_train, target_train_pred))
    CV_metrics['test']['mean_absolute_error'].append(mean_absolute_error(target_test, target_test_pred))
    CV_metrics['training']['median_absolute_error'].append(median_absolute_error(target_train, target_train_pred))
    CV_metrics['test']['median_absolute_error'].append(median_absolute_error(target_test, target_test_pred))
    CV_metrics['training']['mean_squared_error'].append(mean_squared_error(target_train, target_train_pred))
    CV_metrics['test']['mean_squared_error'].append(mean_squared_error(target_test, target_test_pred))
    CV_metrics['training']['mean_squared_error'].append(np.sqrt(mean_squared_error(target_train, target_train_pred)))
    CV_metrics['test']['mean_squared_error'].append(np.sqrt(mean_squared_error(target_test, target_test_pred)))
    CV_metrics['training']['explained_variance_score'].append(explained_variance_score(target_train, target_train_pred))
    CV_metrics['test']['explained_variance_score'].append(explained_variance_score(target_test, target_test_pred))

# learner
NuSVR_API = NuSVR(kernel = 'rbf',
                  C = 1.0,
                  verbose = False,
                  degree = 3,
                  shrinking = True,
                  max_iter = -1,
                  nu = 0.5,
                  tol = 1e-3,
                  cache_size = 200,
                  coef0 = 0.0,
                  gamma = 'auto')

# result: split
StandardScaler_API.fit(data_train)
data_train = StandardScaler_API.transform(data_train)
data_test = StandardScaler_API.transform(data_test)
NuSVR_API.fit(data_train, target_train)
target_train_pred = NuSVR_API.predict(data_train)
target_test_pred = NuSVR_API.predict(data_test)
split_metrics = {'training':{}, 'test':{}}
split_metrics['training']['r2_score'] = r2_score(target_train, target_train_pred)
split_metrics['test']['r2_score'] = r2_score(target_test, target_test_pred)
split_metrics['training']['mean_absolute_error'] = mean_absolute_error(target_train, target_train_pred)
split_metrics['test']['mean_absolute_error'] = mean_absolute_error(target_test, target_test_pred)
split_metrics['training']['median_absolute_error'] = median_absolute_error(target_train, target_train_pred)
split_metrics['test']['median_absolute_error'] = median_absolute_error(target_test, target_test_pred)
split_metrics['training']['mean_squared_error'] = mean_squared_error(target_train, target_train_pred)
split_metrics['test']['mean_squared_error'] = mean_squared_error(target_test, target_test_pred)
split_metrics['training']['mean_squared_error'] = np.sqrt(mean_squared_error(target_train, target_train_pred))
split_metrics['test']['mean_squared_error'] = np.sqrt(mean_squared_error(target_test, target_test_pred))
split_metrics['training']['explained_variance_score'] = explained_variance_score(target_train, target_train_pred)
split_metrics['test']['explained_variance_score'] = explained_variance_score(target_test, target_test_pred)

# result: cross_validation
CV_metrics = {'test': {'r2_score': [], 'mean_absolute_error': [], 'median_absolute_error': [], 'mean_squared_error': [], 'root_mean_squared_error': [], 'explained_variance_score': []}, 'training': {'r2_score': [], 'mean_absolute_error': [], 'median_absolute_error': [], 'mean_squared_error': [], 'root_mean_squared_error': [], 'explained_variance_score': []}}
for train_index, test_index in CV_indices:
    data_train = data.iloc[train_index,:]
    target_train = target.iloc[train_index,:]
    data_test = target.iloc[test_index,:]
    target_test = target.iloc[test_index,:]
    StandardScaler_API.fit(data_train)
    data_train = StandardScaler_API.transform(data_train)
    data_test = StandardScaler_API.transform(data_test)
    NuSVR_API.fit(data_train, target_train)
    target_train_pred = NuSVR_API.predict(data_train)
    target_test_pred = NuSVR_API.predict(data_test)
    CV_metrics['training']['r2_score'].append(r2_score(target_train, target_train_pred))
    CV_metrics['test']['r2_score'].append(r2_score(target_test, target_test_pred))
    CV_metrics['training']['mean_absolute_error'].append(mean_absolute_error(target_train, target_train_pred))
    CV_metrics['test']['mean_absolute_error'].append(mean_absolute_error(target_test, target_test_pred))
    CV_metrics['training']['median_absolute_error'].append(median_absolute_error(target_train, target_train_pred))
    CV_metrics['test']['median_absolute_error'].append(median_absolute_error(target_test, target_test_pred))
    CV_metrics['training']['mean_squared_error'].append(mean_squared_error(target_train, target_train_pred))
    CV_metrics['test']['mean_squared_error'].append(mean_squared_error(target_test, target_test_pred))
    CV_metrics['training']['mean_squared_error'].append(np.sqrt(mean_squared_error(target_train, target_train_pred)))
    CV_metrics['test']['mean_squared_error'].append(np.sqrt(mean_squared_error(target_test, target_test_pred)))
    CV_metrics['training']['explained_variance_score'].append(explained_variance_score(target_train, target_train_pred))
    CV_metrics['test']['explained_variance_score'].append(explained_variance_score(target_test, target_test_pred))

# learner
LinearSVR_API = LinearSVR(loss = 'epsilon_insensitive',
                          C = 1.0,
                          intercept_scaling = 1.0,
                          dual = True,
                          fit_intercept = True,
                          epsilon = 0.0,
                          max_iter = 1000,
                          random_state = None,
                          tol = 1e-4,
                          verbose = 0)

# result: split
StandardScaler_API.fit(data_train)
data_train = StandardScaler_API.transform(data_train)
data_test = StandardScaler_API.transform(data_test)
LinearSVR_API.fit(data_train, target_train)
target_train_pred = LinearSVR_API.predict(data_train)
target_test_pred = LinearSVR_API.predict(data_test)
split_metrics = {'training':{}, 'test':{}}
split_metrics['training']['r2_score'] = r2_score(target_train, target_train_pred)
split_metrics['test']['r2_score'] = r2_score(target_test, target_test_pred)
split_metrics['training']['mean_absolute_error'] = mean_absolute_error(target_train, target_train_pred)
split_metrics['test']['mean_absolute_error'] = mean_absolute_error(target_test, target_test_pred)
split_metrics['training']['median_absolute_error'] = median_absolute_error(target_train, target_train_pred)
split_metrics['test']['median_absolute_error'] = median_absolute_error(target_test, target_test_pred)
split_metrics['training']['mean_squared_error'] = mean_squared_error(target_train, target_train_pred)
split_metrics['test']['mean_squared_error'] = mean_squared_error(target_test, target_test_pred)
split_metrics['training']['mean_squared_error'] = np.sqrt(mean_squared_error(target_train, target_train_pred))
split_metrics['test']['mean_squared_error'] = np.sqrt(mean_squared_error(target_test, target_test_pred))
split_metrics['training']['explained_variance_score'] = explained_variance_score(target_train, target_train_pred)
split_metrics['test']['explained_variance_score'] = explained_variance_score(target_test, target_test_pred)

# result: cross_validation
CV_metrics = {'test': {'r2_score': [], 'mean_absolute_error': [], 'median_absolute_error': [], 'mean_squared_error': [], 'root_mean_squared_error': [], 'explained_variance_score': []}, 'training': {'r2_score': [], 'mean_absolute_error': [], 'median_absolute_error': [], 'mean_squared_error': [], 'root_mean_squared_error': [], 'explained_variance_score': []}}
for train_index, test_index in CV_indices:
    data_train = data.iloc[train_index,:]
    target_train = target.iloc[train_index,:]
    data_test = target.iloc[test_index,:]
    target_test = target.iloc[test_index,:]
    StandardScaler_API.fit(data_train)
    data_train = StandardScaler_API.transform(data_train)
    data_test = StandardScaler_API.transform(data_test)
    LinearSVR_API.fit(data_train, target_train)
    target_train_pred = LinearSVR_API.predict(data_train)
    target_test_pred = LinearSVR_API.predict(data_test)
    CV_metrics['training']['r2_score'].append(r2_score(target_train, target_train_pred))
    CV_metrics['test']['r2_score'].append(r2_score(target_test, target_test_pred))
    CV_metrics['training']['mean_absolute_error'].append(mean_absolute_error(target_train, target_train_pred))
    CV_metrics['test']['mean_absolute_error'].append(mean_absolute_error(target_test, target_test_pred))
    CV_metrics['training']['median_absolute_error'].append(median_absolute_error(target_train, target_train_pred))
    CV_metrics['test']['median_absolute_error'].append(median_absolute_error(target_test, target_test_pred))
    CV_metrics['training']['mean_squared_error'].append(mean_squared_error(target_train, target_train_pred))
    CV_metrics['test']['mean_squared_error'].append(mean_squared_error(target_test, target_test_pred))
    CV_metrics['training']['mean_squared_error'].append(np.sqrt(mean_squared_error(target_train, target_train_pred)))
    CV_metrics['test']['mean_squared_error'].append(np.sqrt(mean_squared_error(target_test, target_test_pred)))
    CV_metrics['training']['explained_variance_score'].append(explained_variance_score(target_train, target_train_pred))
    CV_metrics['test']['explained_variance_score'].append(explained_variance_score(target_test, target_test_pred))
###########################

