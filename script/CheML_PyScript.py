########################### RDKFP
from cheml.chem import RDKFingerprint

RDKFingerprint_API = RDKFingerprint(nBits = 1024,
                                    removeHs = True,
                                    vector = 'bit',
                                    radius = 2,
                                    FPtype = 'Morgan')
RDKFingerprint_API.MolfromFile(molfile = '', path = None, 0,0,...)
data = RDKFingerprint_API.Fingerprint()
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

########################### SupervisedLearning_regression
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import K-fold
from sklearn.preprocessing import StandardScaler
from cheml.nn import nn_psgd
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from cheml.nn import nn_dsgd
from sklearn.svm import SVR


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
trained_network = nn_psgd.train(normalize = False,
                                fit_intercept = True,
                                max_iter = None,
                                random_state = None,
                                tol = 0.001,
                                copy_X = True,
                                alpha = 1.0,
                                solver = 'auto')

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

# learner
nn_dsgd_API = nn_dsgd(normalize = False,
                      fit_intercept = True,
                      max_iter = None,
                      random_state = None,
                      tol = 0.001,
                      copy_X = True,
                      alpha = 1.0,
                      solver = 'auto')

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
###########################

