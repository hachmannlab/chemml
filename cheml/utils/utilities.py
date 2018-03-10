import datetime
import numpy as np
import sys

def list_del_indices(mylist,indices):
    for index in sorted(indices, reverse=True):
        del mylist[index]
    return mylist

def std_datetime_str(mode='datetime'):
    """(std_time_str):
        This function gives out the formatted time as a standard string, i.e., YYYY-MM-DD hh:mm:ss.
    """
    if mode == 'datetime':
        return str(datetime.datetime.now())[:19]
    elif mode == 'date':
        return str(datetime.datetime.now())[:10]
    elif mode == 'time':
        return str(datetime.datetime.now())[11:19]
    elif mode == 'datetime_ms':
        return str(datetime.datetime.now())
    elif mode == 'time_ms':
        return str(datetime.datetime.now())[11:]
    else:
        sys.exit("Invalid mode!")

def slurm_script_exclusive(pyscript_file,nnodes=1,input_slurm_script=None,output_slurm_script='script.slurm'):
    """(slurmjob)
    make the slurm script based on exclusive selection of cores per nodes.

    Parameters
    ----------
    pyscript_file: string
        This is the python script that includes nn_dsgd functions and you are 
        going to run on the cluster. If you are using the cheml python script
        maker this parameter is going to be the name of the final output file.
         
    nnodes: int, optional(default = 1) 
        number of available empty nodes in the cluster.
    
    input_slurm_script: string, optional(default = None)
        The file path to the prepared slurm script. We also locate place of
        --nodes and -np in the script and make sure that provided numbers are 
        equal to number of nodes(nnodes). Also, the exclusive option must be 
        included in the script to have access to an entire node.
    
    output_slurm_script: string, optional(default = 'script.slurm')
        The path and name of the slurm script file that will be saved after 
        changes by this function.
        
    Returns
    -------
    The function will write a slurm script file with the filename passed by 
    output_slurm_script.
    """
    if not input_slurm_script:
        file = ['#!/bin/sh\n', '#SBATCH --time=99:00:00\n', '#SBATCH --job-name="nn"\n', '#SBATCH --output=nn.out\n', '#SBATCH --clusters=chemistry\n', '#SBATCH --partition=beta\n', '#SBATCH --account=pi-hachmann\n', '#SBATCH --exclusive\n', '#SBATCH --nodes=1\n', '\n', '# ====================================================\n', '# For 16-core nodes\n', '# ====================================================\n', '#SBATCH --constraint=CPU-E5-2630v3\n', '#SBATCH --tasks-per-node=1\n', '#SBATCH --mem=64000\n', '\n', '\n', 'echo "SLURM job ID         = "$SLURM_JOB_ID\n', 'echo "Working Dir          = "$SLURM_SUBMIT_DIR\n', 'echo "Temporary scratch    = "$SLURMTMPDIR\n', 'echo "Compute Nodes        = "$SLURM_NODELIST\n', 'echo "Number of Processors = "$SLURM_NPROCS\n', 'echo "Number of Nodes      = "$SLURM_NNODES\n', 'echo "Tasks per Node       = "$TPN\n', 'echo "Memory per Node      = "$SLURM_MEM_PER_NODE\n', '\n', 'ulimit -s unlimited\n', 'module load intel-mpi\n', 'module load python\n', 'module list\n', 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/projects/hachmann/packages/Anaconda:/projects/hachmann/packages/rdkit-Release_2015_03_1:/user/m27/pkg/openbabel/2.3.2/lib\n', 'date\n', '\n', '\n', 'echo "Launch job"\n', 'export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so\n', 'export I_MPI_FABRICS=shm:tcp\n', '\n', 'mpirun -np 2 python test.py\n']
        file[8] = '#SBATCH --nodes=%i\n'%nnodes
        file[-1] = 'mpirun -np %i python %s\n' %(nnodes,pyscript_file)
    else:
        file = open(input_slurm_script,'r')
        file = file.readlines()
        exclusive_flag = False
        nodes_flag = False
        np_flag = False
        for i,line in enumerate(file):
            if '--exclusive' in line:
                exclusive_flag = True
            elif '--nodes' in line:
                nodes_flag = True
                ind = line.index('--nodes')
                file[i] = line[:ind]+'--nodes=%i\n'%nnodes
            elif '-np' in line:
                np_flag = True
                ind = line.index('--nodes')
                file[i] = line[:ind]+'--nodes=%i\n'%nnodes                     
        if not exclusive_flag:
            file = file[0] + ['#SBATCH --exclusive\n'] + file[1:]
            msg = "The --exclusive option is not available in the slurm script. We added '#SBATCH --exclusive' to the first of file."
            warnings.warn(msg,UserWarning)
        if not nodes_flag:
            file = file[0] + ['#SBATCH --nodes=%i\n'%nnodes] + file[1:]
            msg = "The --nodes option is not available in the slurm script. We added '#SBATCH --nodes=%i' to the first of file."%nnodes
            warnings.warn(msg,UserWarning)
        if not np_flag:
            file.append('mpirun -np %i python %s\n'%(nnodes,pyscript_file))
            msg = "The -np option is not available in the slurm script. We added 'mpirun -np %i python %s'to the end of file."%(nnodes,pyscript_file) 
            warnings.warn(msg,UserWarning)
            
    script = open(output_slurm_script,'w')
    for line in file:    
        script.write(line)
    script.close()

def chunk(xs, n, X=None, Y=None):
    """
    X and Y must be np array
    the term of use:
                it = chunk ( range(len(X), n, X, Y)
                X_chunk, y_chunk = it.next()
    n is the number of chunks (#total_batch).
    """
    ys = list(xs)
    # random.shuffle(ys)
    size = len(ys) // n
    leftovers= ys[size*n:]
    for c in xrange(n):
        if leftovers:
           extra= [ leftovers.pop() ] 
        else:
           extra= []
        if isinstance(X,np.ndarray):
            if isinstance(Y, np.ndarray):
                yield X[ys[c*size:(c+1)*size] + extra], Y[ys[c*size:(c+1)*size] + extra]
            else:
                yield X[ys[c * size:(c + 1) * size] + extra]
        else:
            yield ys[c*size:(c+1)*size] + extra

def choice(X, Y=None, n=0.1, replace=False):
    """
    Generates a random sample from a given 1-D array. Bassicaly same as np.random.choice with pre- and post-processing
     steps. Sampling without replacement.
    :param X: 1-D array-like
        A random sample will be generated from its elements.
    :param Y: 1-D array-like, optional (default = None)
        A random sample will be generated from its elements.
    :param n: int or float between zero and one, optional (default = 0.1)
        size of sample
    :param replace: boolean, default=False
        whether the sample is with or without replacement
    :return: a_out: 1-D array-like
                the array of out of sample elements
    :return: a_sample: 1-D array-like, shape (size,)
                the sample array
    """
    if not isinstance(n,int):
            n = int(n*len(X))
    ind_sample = np.random.choice(len(X),n,replace=replace)
    ind_out = np.array([i for i in xrange(len(X)) if i not in ind_sample])
    X_sample = X[ind_sample]
    X_out = X[ind_out]
    if isinstance(Y,np.ndarray):
        if len(Y) != len(X):
            raise Exception('X and Y must be same size')
        Y_sample = Y[ind_sample]
        Y_out = Y[ind_out]
    else:
        Y_sample = None
        Y_out = None
    return X_out, X_sample, Y_out, Y_sample

def return2Dshape(shape):
    if len(shape) == 2:
        return shape
    elif len(shape) == 1:
        return (shape[0],None)
    else:
        raise Exception('input dimension is greater than 2')

def bool_formatter(bool):
    if bool:
        return("true")
    else:
        return("false")
