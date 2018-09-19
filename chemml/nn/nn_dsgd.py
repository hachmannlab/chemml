import numpy as np
from mpi4py import MPI
import warnings
import multiprocessing
import nn_psgd
from ..utils import chunk

def train(X,Y,nneurons,input_act_funcs,validation_size=0.2,learn_rate=0.001,rms_decay=0.9,n_epochs=10000,
    batch_size=256,n_hist=20,n_check=50,threshold=0.1, print_level=1):
    """
    Main distributed memory function
    
    Parameters
    ----------
    All available parameters for nn_psgd - n_cores
    The number of cores will be directly passed to the mpirun command
            
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
            if i!=0:
                comm.send(X[indices],dest=i, tag = 7)
                comm.send(Y[indices],dest=i, tag = 77)
            else:
                Xnew = X[indices]
                Ynew = Y[indices]
        X = Xnew
        Y = Ynew
    else:
        X = comm.recv(source=0, tag = 7)
        Y = comm.recv(source=0, tag = 77)

    trained_network =  nn_psgd.train(X,Y,nneurons=nneurons,
    input_act_funcs=input_act_funcs,learn_rate=learn_rate,rms_decay=rms_decay,
    n_epochs=n_epochs,batch_size=batch_size,n_cores=multiprocessing.cpu_count(),n_hist=n_hist,
    n_check=n_check,threshold=threshold, print_level=print_level)
    
    trained_network = comm.gather(trained_network,root=0)
    if rank==0:
        return trained_network

def output(X,nnets):
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
    #MPI
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    if rank == 0:
        results = []
        for nn in nnets:
            results+= [nn_psgd._output(X,nn['weights'],nn_psgd.act_funcs_from_string(nn['act_funcs'],len(nn['weights'])-1))]
        return results