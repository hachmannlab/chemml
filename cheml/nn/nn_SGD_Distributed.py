import numpy as np
from mpi4py import MPI
import warnings
import multiprocessing
import nn_psgd
from ..utils import chunk

def nn_dsgd_train(X,Y,nneurons,input_act_funcs,validation=0.1,learn_rate=0.001,rms_decay=0.9,n_epochs=10000,
    batch_size=256,n_cores=1,n_hist=20,n_check=50,threshold=0.1, print_level=1):
    """
    Main distributed memory function
    
    Parameters
    ----------
    All available parameters for nn_psgd + validation
    
    validation: float between zero and one, optional(default = 0.1)
        The ratio of data set to be used for validation of trained network. The 
        rest of data points will be used as training data.
            
    Returns
    -------
    trained_network: a list of dicts with trained weights and the activation functions from
    each node

    """

    # MPI
    comm=MPI.COMM_WORLD
    rank=comm.rank
    size=comm.size
    cpu_count = multiprocessing.cpu_count()
    cpu_count = comm.gather(cpu_count,root=0)
    if rank == 0:
        N = len(X)
        n_cores = sum(cpu_count)
        chunk_list= list( chunk(range(N),n_cores) )
        indices =[]
        for i,c in enumerate(cpu_count):
            indices = []
            for j in range(c):
                indices+=chunk_list.pop()
            comm.send(indices,dest=i)
    
    indices = comm.recv(source=0)
    cut_ind = int(validation * len(indices))
    X_validation = X[indices[:cut_ind]]
    Y_validation = Y[indices[:cut_ind]]
    X = X[indices[cut_ind:]]
    Y = Y[indices[cut_ind:]]
    trained_network =  nn_psgd.train(X,X_validation,Y,Y_validation,nneurons=nneurons,
    input_act_funcs=input_act_funcs,learn_rate=learn_rate,rms_decay=rms_decay,
    n_epochs=n_epochs,batch_size=batch_size,n_cores=n_cores,n_hist=n_hist,
    n_check=n_check,threshold=threshold, print_level=print_level)
    trained_network = comm.gather(trained_network,root=0)
    if rank==0:
        return trained_network

def nn_dsgd_output(X,nnets):
    """(nn_dsgd_output)
    User accessible output for neural network given trained weights.
    
    Parameters
    ----------
        X: array
            Input features
        
        nnets: list of dict
            A list of neural networks from each cluster. keys required weights and
             activation functions
    Returns
    -------
    predicted values in array type
    """
    if rank == 0:
        for nn in nnets:
            return nn_psgd._output(X,nn['weights'],nn_psgd.act_funcs_from_string(nn['act_funcs'],len(nn['weights'])-1))
