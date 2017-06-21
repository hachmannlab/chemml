import numpy as np
import pandas as pd
from scipy import special as spspecial
from multiprocessing import Pool, sharedctypes, current_process
# from ..utils import check_input

def check_input(X,name,n0=None,n1=None, format_out='df'):
    """
    Makes sure that input is a 2-D numpy array or pandas dataframe in the correct format.

    :param X: numpy.ndarray or pandas.DataFrame
        input data
    :param name: string
        name of input (e.g. training input)
    :param n0: int
        number of data entries
    :param n1: int
        number of features
    :param format_out: string ('df' or 'ar'), optional (default = 'df')

    :return input data converted to array or dataframe
    :return the header of dataframe
        if input data is not a dataframe return None

    """
    if not (X.ndim == 2):
        raise Exception(name+' needs to be two dimensional')
    if isinstance(X, pd.DataFrame):
        if format_out == 'ar':
            if X.shape[1]>1:
                header = X.columns
                X = X.values
            else:
                if n0 == 1:
                    header = X.columns
                    X = X[header[0]].values
                else:
                    header = X.columns
                    X = X.values
        else:
            header = X.columns
        if not np.can_cast(X.dtypes, np.float, casting='same_kind'):
            raise Exception(name + ' cannot be cast to floats')
    elif isinstance(X, np.ndarray):
        if format_out == 'df':
            X = pd.DataFrame(X)
            header = None
        else:
            header = None
    else:
        raise Exception(name+' needs to be either pandas dataframe or numpy array')
    if n0 and X.shape[0] != n0:
        raise Exception(name+' has an invalid number of data entries')
    if n1 and X.shape[1] != n1:
        raise Exception(name+' has an invalid number of feature entries')
    return X.astype(float), header

def np_to_c(np_array):
    """
    Converts numpy array to c-type to allow for sharing across different threads
    :param np_array:
    :return: ctype array
    """
    tmp = np.ctypeslib.as_ctypes(np_array)
    c_array = sharedctypes.Array(tmp._type_, tmp, lock=False)
    return c_array

def initialize_data(np_data):
    """
    Initializes training and testing data
    :param np_data: numpy array

    :return: tuple of c-type arrays for data
    """
    c_data = np_to_c(np_data)
    return c_data

def train_init(c_X, c_Y, model, evaluator,kfold):
    global X
    X = np.ctypeslib.as_array(X)
    global Y
    Y = np.ctypeslib.as_array(Y)
    global Model
    Model = model
    global Evaluator
    Evaluator = evaluator

def train(indices):
    train_index, test_index = indices
    api, trainer, predicter = model
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    api = trainer(api,X_train,Y_train)
    Y_pred = predicter(api,X_test)
    results = {metric: evaluator[metric](Y_test, Y_pred) for metric in evaluator}
    print current_process(), ':', results

def main(X,Y,kfold,model,evaluator):
    X, _ = check_input(X,'input data', format_out='ar')
    Y, _ = check_input(Y,'output data',n0=X.shape[0],format_out='ar')
    N = X.shape[0]
    n_processes = kfold.get_n_splits(X)
    print '******************',n_processes

    c_X = initialize_data(X)
    c_Y = initialize_data(Y)
    del X
    del Y

    # for train_index, test_index in kfold.split(X):
    #     X_train, X_test = X[train_index], X[test_index]
    #     y_train, y_test = y[train_index], y[test_index]
    pool = Pool(processes=n_processes, initializer=train, initargs=(c_X, c_Y, model, evaluator,kfold))
    indices = [(initialize_data(train_index),initialize_data(test_index)) for train_index, test_index in kfold.split(xrange(N))]
    print '******************', len(indices)
    pool.map(train, indices)

if __name__ == '__main__':
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5)
    N=2000
    X = (np.random.rand(N, 1) - 0.5) * 10
    Y = np.sin(X)
    model = lambda X,t: np.sin(X)+0.01*t
    evaluator = {'fake': lambda y,yy: np.mean(y-yy)}
    main(X,Y,kf,model,evaluator)
