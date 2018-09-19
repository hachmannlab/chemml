import numpy as np
import pandas as pd
from multiprocessing import Pool, sharedctypes, cpu_count
from ..utils import check_input
import time

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

def _search_init(c_X, c_Y):
    global X
    X = np.ctypeslib.as_array(c_X)
    global Y
    Y = np.ctypeslib.as_array(c_Y)

def _log_results(result):
    All_results.append(result)

def search(args):
    train_index, test_index,i = args
    print i
    api, trainer, predicter = Model
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    api = trainer(api,X_train,Y_train)
    Y_pred = predicter(api, X_test)
    result = {metric: Evaluator[metric](Y_test, Y_pred) for metric in evaluator}
    return result

def fit(X,Y,kfold,model, evaluator, n_cores=1):
    """
    The fit function to start the search.

    :param X: input data
    :param Y: output data
    :param kfold: kfold cross validation from sklearn
    :param model: tuple with length 3
        ML model consists of (api, trainer(api,X,Y), predicter(api,X))
    :param evaluator: dictionary
        {'name of evaluation method' : function(Y,Y_pred)}
    :param n_cores: int, default=1
        determines the number of process for the multiprocessing task. If None the
    :return: results, dictionary
        {'name of evaluation method' : evaluation score}
    """
    global All_results
    All_results = []
    global Model
    Model = model
    global Evaluator
    Evaluator = evaluator

    X, _ = check_input(X,'input data', format_out='ar')
    Y, _ = check_input(Y,'output data',n0=X.shape[0],format_out='ar')
    N = X.shape[0]
    # n_processes = kfold.get_n_splits(X)
    print 'number of available CPUs: %i' % cpu_count()
    print 'requested number of processes: %i' %n_cores
    c_X = initialize_data(X)
    c_Y = initialize_data(Y)
    del X
    del Y
    pool = Pool(processes=n_cores, initializer=_search_init, initargs=(c_X, c_Y))
    # indices = [(initialize_data(train_index),initialize_data(test_index)) for train_index, test_index in kfold.split(xrange(N))]
    # pool.map(train, indices)
    index_generator = kfold.split(xrange(N))
    for i in range(kfold.get_n_splits()):
        train_index, test_index = index_generator.next()
        pool.apply_async(search, args = ((train_index, test_index,i),), callback = _log_results)
    pool.close()
    pool.join()
    return All_results

def tot_exec_time_str(time_start):
    """(tot_exec_time_str):
        This function gives out the formatted time string.
    """
    time_end = time.time()
    exec_time = time_end-time_start
    tmp_str = "execution time: %0.2fs (%dh %dm %0.2fs)" %(exec_time, exec_time/3600, (exec_time%3600)/60,(exec_time%3600)%60)
    return tmp_str

if __name__ == '__main__':
    from sklearn.model_selection import KFold

    start_time = time.time()
    kf = KFold(n_splits=4)
    N=2000
    X = (np.random.rand(N, 1) - 0.5) * 10
    Y = np.sin(X)
    model = (7,lambda api,X,Y: len(X)+len(Y),lambda api,X: len(X))
    evaluator = {'fake': lambda y,yy: len(y) - yy}
    print fit(X,Y,kf,model, evaluator, n_cores=4)
    print tot_exec_time_str(start_time)