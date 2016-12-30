import numpy as np
from scipy import special as spspecial
import sys
from multiprocessing import Pool, sharedctypes, Value
import copy

def act_funcs_from_string(input_act_funcs, n):
    """
    translate activation functions input from user
    Inputs:
    input_act_funcs - list of activation functions or just one function. should be same
    length as nneurons if a list; allowed functions as of Jan 2015 are
          * tanh
          * logistic
    n - number of activation functions
    Outputs:
    list of activation functions to be used and their gradients

    """
    act_func_dict = {'tanh':[np.tanh,lambda x: np.power(np.cosh(x),-2)],
        'logistic':[spspecial.expit,lambda x: spspecial.expit(x) * (1- spspecial.expit(x))]}
    if isinstance(input_act_funcs,list):
        if len(input_act_funcs) != n:
            raise Exception('List of activation function is of different \
                length from list of neuron numbers')
        else:
            return map(lambda x: act_func_dict[x],input_act_funcs)
    else:
        return [act_func_dict[input_act_funcs]]*n

def check_array(X,name,n0=None,n1=None):
    """
    Makes sure that array is correct
    Inputs:
    X - array to be checked
    n0 - number of data entries
    n1 - number of features
    name - name of array (e.g. training input)
    Output:
    array converted to floats
    """
    if not (isinstance(X, np.ndarray) and X.ndim == 2):
        raise Exception(name+' needs to be two dimensional numpy array')
    elif not np.can_cast(X.dtype,np.float,casting='same_kind'):
        raise Exception(name+' cannot be cast to floats')
    if n0 and X.shape[0] != n0:
        raise Exception(name+' has an invalid number of data entries')
    if n1 and X.shape[1] != n1:
        raise Exception(name+' has an invalid number of feature entries')
    return X.astype(float)

def np_to_c(np_array):
    """
    Converts numpy array to c-type to allow for sharing across different threads
    :param np_array:
    :return: ctype array
    """
    tmp = np.ctypeslib.as_ctypes(np_array)
    c_array = sharedctypes.Array(tmp._type_, tmp, lock=False)
    return c_array

def initialize_weights(complete_nneurons,n_hist):
    """
    Initialize weights randomly using numpy's random number generator
    Inputs:
    complete_nneurons - list of number of neurons (including input and output layers)
    Output:
    list of initialized weights
    """
    weights = []
    for prev_nneurons,nneurons in zip(complete_nneurons[:-1],complete_nneurons[1:]):
        weight = np_to_c(np.random.rand(prev_nneurons,nneurons) - 0.5)
        bias = np_to_c(np.random.rand(nneurons,1) - 0.5)
        weights.append([weight,bias])

    nphist = np.empty((n_hist,complete_nneurons[-1]+1))
    nphist[:,:-1] = np.inf
    nphist[:,-1] = 0
    return weights,np_to_c(nphist)

def initialize_data(np_X_train,np_Y_train,np_X_test,np_Y_test):
    """
    Initializes training and testing data
    :param np_X_train: numpy array
    :param np_Y_train: numpy array
    :param np_X_test: numpy array
    :param np_Y_test: numpy array
    :return: tuple of c-type arrays for features_training, targets_training, features_testing, targets_testing
    """
    c_X_train = np_to_c(np_X_train)
    c_Y_train = np_to_c(np_Y_train)
    c_X_test = np_to_c(np_X_test)
    c_Y_test = np_to_c(np_Y_test)
    return c_X_train,c_Y_train,c_X_test,c_Y_test

def feed_forward(X,weights,act_funcs):
    """
    Feed forward through neural network
    Inputs:
    X - input
    weights - weights
    act_funcs - activation functions
    Output:
    his - output from each layer of the network
    dhis - h'(a) for each layer in the network, where h' is the derivative of the activation
    function, and a is the sum of weighted outputs from the previous layer for use in back propagation

    """

    his = [X]  # outputs of neurons at all layers, initialized to input layer
    dhis = [np.ones((X.shape[0],X.shape[1]))]
    hi = his[0]  # output of input layer, de-reffed for efficiency
    for i, [weight, act_func] in enumerate(zip(weights[:-1], act_funcs)):
        arg = np.dot(hi, weight[0]) + weight[1].T  # input to activation function
        hi = act_func[0](arg)  # output at layer i of network
        dhi = act_func[1](arg)  # derivative of hi wrt arg
        his.append(hi)
        dhis.append(dhi)
    his.append(np.dot(his[-1], weights[-1][0]) + weights[-1][1].T)
    return his, dhis

def _output(X, weights, act_funcs):
    """
    Internal function. Gives the output of the network without book keeping each layer
    :param X: Inputs
    :param weights: type list of lists : weights
    :param act_funcs: type list : activation functions
    :return: type array : output of network

    """

    hi = X
    for weight,act_func in zip(weights[:-1],act_funcs):
        hi = act_func[0](np.dot(hi,weight[0]) + weight[1].T)
    return np.dot(hi, weights[-1][0]) + weights[-1][1].T

def output(X,nn):
    """
    User accessible output for neural network given trained weights.
    :param X: type array:Input features
    :param nn: type dict : Neural network: keys required weights and activation functions
    :return: type array: output values
    """
    return _output(X,nn['weights'],act_funcs_from_string(nn['act_funcs'],len(nn['weights'])-1))

def _c_to_numpy_weights(cweights):
    """
    Convenience function to convert shared ctype weights to numpy weights
    :param cweights:
    :return: type list of lists: lists of weights and biases of each layer.
    :note: This still shares memory with ctype weights
    """
    weights = []
    for cweight,cbias in cweights:
        weight = np.ctypeslib.as_array(cweight)
        bias = np.ctypeslib.as_array(cbias)
        weights.append([weight,bias])
    return weights

def back_prop_init(c_X_train,c_Y_train,c_X_test,c_Y_test,shared_weights,shared_hist,input_act_funcs):
    """
    Initializing back propagation.  Initializes global vars for each thread, and makes sure that all the
    threads share memory to save memory.
    :param c_X_train: ctype array of the training features
    :param c_Y_train: ctype array of the training targets
    :param c_X_test: ctype array of the testing features
    :param c_Y_test: ctype array of the testing targets
    :param shared_weights: list of weights
    :param shared_hist: array : history for early stopping
    :param input_act_funcs: user input of activation functions
    :return:
    """
    # TODO: sanitize activation function input

    global weights
    weights = _c_to_numpy_weights(shared_weights)
    global act_funcs
    act_funcs = act_funcs_from_string(input_act_funcs,len(weights)-1)
    global hist
    hist = np.ctypeslib.as_array(shared_hist)
    global X_train
    X_train = np.ctypeslib.as_array(c_X_train)
    global Y_train
    Y_train = np.ctypeslib.as_array(c_Y_train)
    global X_test
    X_test = np.ctypeslib.as_array(c_X_test)
    global Y_test
    Y_test = np.ctypeslib.as_array(c_Y_test)

def initialize_rms_weights(weights):
    """
    Initializes weights for rms prop
    :param weights: type list of lists: weights in each layer
    :return: zeroed list of lists with the same dimensions as the input list of lists
    """
    rms_weights = copy.deepcopy(weights)
    for weight,bias in rms_weights:
        weight[:] = 0
        bias[:] = 0
    return rms_weights

def prevent_underflow(np_array):
    """
    Stops underflow by replacing 0 with smallest numpy value
    :param np_array:
    :return:
    """
    zero_inds = (np_array == 0)
    if zero_inds.any():
        np_array[zero_inds] = float(np.finfo(np.float64).tiny)
    return np_array

def back_propagate(para_input):
    """
    Performs back propagation on one thread to train the neural network
    :param para_input: type list: input parameters (unpacking needed due to python's handling)
    """
    pool_index, train_start, train_end, learn_rate, rms_decay, n_epochs, \
        batch_size, n_cores, n_hist, n_check, threshold, print_level = para_input
    hist_count = 0
    rms_weights = initialize_rms_weights(weights)

    for i in range(n_epochs):
        if i % n_check == 0:
            cur_err = np.mean(np.power(_output(X_test, weights, act_funcs) - Y_test, 2), axis=0)
            if print_level > 0:
                print i, cur_err

            if np.mean(hist[:, -1]) >= threshold:
                break
            else:
                hist_index = hist_count * n_cores + pool_index
                hist[hist_index, :-1] = cur_err
                if np.mean(np.median(hist, axis=0)[:-1] - cur_err) <= 0:
                    hist[hist_index, -1] = 1
                else:
                    hist[hist_index, -1] = 0
                hist_count = (hist_count + 1) % n_hist

        for start_ind in range(train_start, train_end-1, batch_size):
            X_train_batch = X_train[start_ind:start_ind+batch_size]
            Y_train_batch = Y_train[start_ind:start_ind+batch_size]

            his,dhis = feed_forward(X_train_batch,weights,act_funcs)
            di = his[-1] - Y_train_batch

            for weight,hi,dhi,rms_weight in zip(weights[::-1],his[:-1][::-1],dhis[::-1],rms_weights[::-1]):
                weight_grad = np.dot(hi.T,di)/di.shape[0]
                bias_grad = np.mean(di,axis=0)[:,None]
                rms_weight[0] = prevent_underflow((1 - rms_decay) * np.power(weight_grad,2) + rms_decay * rms_weight[0])
                rms_weight[1] = prevent_underflow((1 - rms_decay) * np.power(bias_grad,2) + rms_decay * rms_weight[1])
                weight[0] -= learn_rate * weight_grad / np.power(rms_weight[0],0.5)
                weight[1] -= learn_rate * bias_grad / np.power(rms_weight[1],0.5)
                if np.isnan(weight[0]).any() or np.isinf(weight[0]).any() or np.isnan(weight[1]).any() \
                    or np.isinf(weight[1]).any():
                    break
                di = dhi * np.dot(di,weight[0].T)
ZZ

# @profile
def train(X_train,X_test,Y_train,Y_test,nneurons,input_act_funcs,learn_rate=0.001,rms_decay=0.9,n_epochs=10000,
    batch_size=256,n_cores=1,n_hist=20,n_check=50,threshold=0.1, print_level=1):
    """
    Main training function
    Inputs:
    X_train - input training data
    X_test - input testing data
    Y_train - output training data
    Y_test - output testing data
    nneurons - list of integers describing how many neurons there are in each layer
    input_act_funcs - list of activation functions or just one function. should be same
    length as nneurons if a list
    Outputs:
    nnet - a tuple with trained weights and the activation functions

    """
    X_train = check_array(X_train,'Training input')
    X_test = check_array(X_test,'Testing input',n1=X_train.shape[1])
    Y_train = check_array(Y_train,'Training output',n0=X_train.shape[0])
    Y_test = check_array(Y_test,'Testing output',n0=X_test.shape[0],n1=Y_train.shape[1])
    n_features = X_train.shape[1]
    n_outputs = Y_train.shape[1]
    N = X_train.shape[0]

    c_X_train, c_Y_train, c_X_test, c_Y_test = initialize_data(X_train,Y_train,X_test,Y_test)
    del X_train
    del X_test
    del Y_train
    del Y_test

    cweights, chist = initialize_weights([n_features] + nneurons +
        [n_outputs],n_hist*n_cores)

    big_batch_size = (N / n_cores) + 1
    para_inputs = [[i, j, j+big_batch_size,learn_rate,rms_decay,n_epochs,batch_size,n_cores,n_hist,
        n_check,threshold, print_level] for i,j in enumerate(range(0,N,big_batch_size))]
    pool = Pool(processes=n_cores, initializer=back_prop_init, initargs=(c_X_train,c_Y_train,c_X_test,c_Y_test,
        cweights, chist, input_act_funcs))
    pool.map(back_propagate, para_inputs)

    # use this to test individual thread
    # back_prop_init(c_X_train,c_Y_train,c_X_test,c_Y_test,cweights, chist, input_act_funcs)
    # back_propagate([0,0,N,learn_rate,rms_decay,n_epochs,batch_size,1,n_hist,n_check,threshold])

    final_weights = _c_to_numpy_weights(cweights)
    return {'weights':final_weights,'act_funcs':input_act_funcs}

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    N = 2000
    X = (np.random.rand(N,1) - 0.5) * 10
    Y = np.sin(X)
    X_train = X[:N/2]
    X_test = X[N/2:]
    Y_train = Y[:N/2]
    Y_test = Y[N/2:]
    nnet = train(X_train,X_test,Y_train,Y_test,[20,10],'tanh',learn_rate=0.001,n_epochs=50000,n_cores=2)
    print np.mean(np.abs(output(X_test,nnet) - Y_test),axis=0)
    plt.plot(X_test[:,0],output(X_test,nnet)[:,0],'x')
    plt.plot(X_test[:,0],Y_test[:,0],'o')
    plt.show()
