from builtins import range
import datetime
import numpy as np
import time
import os


def list_del_indices(mylist,indices):
    """
    iteratively remove elements of a list by indices
    Parameters
    ----------
    mylist : list
        the list of elements of interest

    indices : list
        the list of indices of elements that should be removed

    Returns
    -------
    list
        the reduced mylist entry

    """
    for index in sorted(indices, reverse=True):
        del mylist[index]
    return mylist


def std_datetime_str(mode='datetime'):
    """ human readable data and time
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
        msg = 'The mode value must be one of datetime, date, time, datetime_ms, or time_ms.'
        raise ValueError(msg)


def tot_exec_time_str(time_start):
    """ execution time
        This function gives out the formatted time string.
    """
    time_end = time.time()
    exec_time = time_end-time_start
    tmp_str = "execution time: %0.2fs (%dh %dm %0.2fs)" %(exec_time, exec_time/3600, (exec_time%3600)/60,(exec_time%3600)%60)
    return tmp_str


# def slurm_script_exclusive(pyscript_file,nnodes=1,input_slurm_script=None,output_slurm_script='script.slurm'):
#     """(slurmjob)
#     make the slurm script based on exclusive selection of cores per nodes.
#
#     Parameters
#     ----------
#     pyscript_file: string
#         This is the python script that includes nn_dsgd functions and you are
#         going to run on the cluster. If you are using the chemml python script
#         maker this parameter is going to be the name of the final output file.
#
#     nnodes: int, optional(default = 1)
#         number of available empty nodes in the cluster.
#
#     input_slurm_script: string, optional(default = None)
#         The file path to the prepared slurm script. We also locate place of
#         --nodes and -np in the script and make sure that provided numbers are
#         equal to number of nodes(nnodes). Also, the exclusive option must be
#         included in the script to have access to an entire node.
#
#     output_slurm_script: string, optional(default = 'script.slurm')
#         The path and name of the slurm script file that will be saved after
#         changes by this function.
#
#     Returns
#     -------
#     The function will write a slurm script file with the filename passed by
#     output_slurm_script.
#     """
#     if not input_slurm_script:
#         file = ['#!/bin/sh\n', '#SBATCH --time=99:00:00\n', '#SBATCH --job-name="nn"\n', '#SBATCH --output=nn.out\n', '#SBATCH --clusters=chemistry\n', '#SBATCH --partition=beta\n', '#SBATCH --account=pi-hachmann\n', '#SBATCH --exclusive\n', '#SBATCH --nodes=1\n', '\n', '# ====================================================\n', '# For 16-core nodes\n', '# ====================================================\n', '#SBATCH --constraint=CPU-E5-2630v3\n', '#SBATCH --tasks-per-node=1\n', '#SBATCH --mem=64000\n', '\n', '\n', 'echo "SLURM job ID         = "$SLURM_JOB_ID\n', 'echo "Working Dir          = "$SLURM_SUBMIT_DIR\n', 'echo "Temporary scratch    = "$SLURMTMPDIR\n', 'echo "Compute Nodes        = "$SLURM_NODELIST\n', 'echo "Number of Processors = "$SLURM_NPROCS\n', 'echo "Number of Nodes      = "$SLURM_NNODES\n', 'echo "Tasks per Node       = "$TPN\n', 'echo "Memory per Node      = "$SLURM_MEM_PER_NODE\n', '\n', 'ulimit -s unlimited\n', 'module load intel-mpi\n', 'module load python\n', 'module list\n', 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/projects/hachmann/packages/Anaconda:/projects/hachmann/packages/rdkit-Release_2015_03_1:/user/m27/pkg/openbabel/2.3.2/lib\n', 'date\n', '\n', '\n', 'echo "Launch job"\n', 'export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so\n', 'export I_MPI_FABRICS=shm:tcp\n', '\n', 'mpirun -np 2 python test.py\n']
#         file[8] = '#SBATCH --nodes=%i\n'%nnodes
#         file[-1] = 'mpirun -np %i python %s\n' %(nnodes,pyscript_file)
#     else:
#         file = open(input_slurm_script,'r')
#         file = file.readlines()
#         exclusive_flag = False
#         nodes_flag = False
#         np_flag = False
#         for i,line in enumerate(file):
#             if '--exclusive' in line:
#                 exclusive_flag = True
#             elif '--nodes' in line:
#                 nodes_flag = True
#                 ind = line.index('--nodes')
#                 file[i] = line[:ind]+'--nodes=%i\n'%nnodes
#             elif '-np' in line:
#                 np_flag = True
#                 ind = line.index('--nodes')
#                 file[i] = line[:ind]+'--nodes=%i\n'%nnodes
#         if not exclusive_flag:
#             file = file[0] + ['#SBATCH --exclusive\n'] + file[1:]
#             msg = "The --exclusive option is not available in the slurm script. We added '#SBATCH --exclusive' to the first of file."
#             warnings.warn(msg,UserWarning)
#         if not nodes_flag:
#             file = file[0] + ['#SBATCH --nodes=%i\n'%nnodes] + file[1:]
#             msg = "The --nodes option is not available in the slurm script. We added '#SBATCH --nodes=%i' to the first of file."%nnodes
#             warnings.warn(msg,UserWarning)
#         if not np_flag:
#             file.append('mpirun -np %i python %s\n'%(nnodes,pyscript_file))
#             msg = "The -np option is not available in the slurm script. We added 'mpirun -np %i python %s'to the end of file."%(nnodes,pyscript_file)
#             warnings.warn(msg,UserWarning)
#
#     script = open(output_slurm_script,'w')
#     for line in file:
#         script.write(line)
#     script.close()


def chunk(xs, n, X=None, Y=None):
    """
    X and Y must be numpy array
    n is the number of chunks (#total_batch).

    Examples
    --------
    it = chunk ( range(len(X), n, X, Y)
    X_chunk, y_chunk = next(it)

    """
    ys = list(xs)
    # random.shuffle(ys)
    size = len(ys) // n
    leftovers= ys[size*n:]
    for c in range(n):
        if leftovers:
           extra = [ leftovers.pop() ]
        else:
           extra = []
        if isinstance(X,np.ndarray):
            if isinstance(Y, np.ndarray):
                yield X[ys[c*size:(c+1)*size] + extra], Y[ys[c*size:(c+1)*size] + extra]
            else:
                yield X[ys[c * size:(c + 1) * size] + extra]
        else:
            yield ys[c*size:(c+1)*size] + extra


# def choice(X, Y=None, n=0.1, replace=False):
#     """
#     Generates a random sample from a given 1-D array. Bassicaly same as np.random.choice with pre- and post-processing
#      steps. Sampling without replacement.
#     :param X: 1-D array-like
#         A random sample will be generated from its elements.
#     :param Y: 1-D array-like, optional (default = None)
#         A random sample will be generated from its elements.
#     :param n: int or float between zero and one, optional (default = 0.1)
#         size of sample
#     :param replace: boolean, default=False
#         whether the sample is with or without replacement
#     :return: a_out: 1-D array-like
#                 the array of out of sample elements
#     :return: a_sample: 1-D array-like, shape (size,)
#                 the sample array
#     """
#     if not isinstance(n,int):
#             n = int(n*len(X))
#     ind_sample = np.random.choice(len(X),n,replace=replace)
#     ind_out = np.array([i for i in xrange(len(X)) if i not in ind_sample])
#     X_sample = X[ind_sample]
#     X_out = X[ind_out]
#     if isinstance(Y,np.ndarray):
#         if len(Y) != len(X):
#             raise Exception('X and Y must be same size')
#         Y_sample = Y[ind_sample]
#         Y_out = Y[ind_out]
#     else:
#         Y_sample = None
#         Y_out = None
#     return X_out, X_sample, Y_out, Y_sample
#
# def return2Dshape(shape):
#     if len(shape) == 2:
#         return shape
#     elif len(shape) == 1:
#         return (shape[0],None)
#     else:
#         raise Exception('input dimension is greater than 2')


def bool_formatter(bool_value):
    """
    convert Python boolean to json/xml format of boolean

    Parameters
    ----------
    bool_value: bool
        the boolean value that needs to be converted

    Returns
    -------
    str
        either "true" or "false"

    """
    if isinstance(bool_value, bool):
        if bool_value:
            return("true")
        else:
            return("false")
    else:
        msg = "bool_value must be a boolean"
        raise ValueError(msg)


def padaxis(array, new_size, axis, pad_value=0, pad_right=True):
    """
    Padds one axis of an array to a new size

    This is just a wrapper for np.pad, more usefull when only padding a single axis

    Parameters
    ----------
    array: array
        the array to pad

    new_size: int
        the new size of the specified axis

    axis: int
        axis along which to pad

    pad_value: float or int, optional(default=0)
        pad value

    pad_right: bool, optional(default=True)
        if True pad on the right side, otherwise pad on left side

    Returns
    -------
        padded_array: np.array

    """
    add_size = new_size - array.shape[axis]
    assert add_size >= 0, 'Cannot pad dimension {0} of size {1} to smaller size {2}'.format(axis, array.shape[axis], new_size)
    pad_width = [(0,0)]*len(array.shape)

    #pad after if int is provided
    if pad_right:
        pad_width[axis] = (0, add_size)
    else:
        pad_width[axis] = (add_size, 0)

    return np.pad(array, pad_width=pad_width, mode='constant', constant_values=pad_value)


def mol_shapes_to_dims(mol_tensors=None, mol_shapes=None):
    ''' 
    Helper function, returns dim sizes for molecule tensors given tensors or
    tensor shapes

    Parameters
    ----------
    mol_tensors: tensorflow.tensor, default=None
        tensor of molecule
    
    mol_shapes: tuple, default=None
        shape of the molecule tensor

    Returns
    -------
    max_atoms1
        maximum number of atoms

    max_degree1
        maximum degree

    num_atom_features
        total features

    num_bond_features
        total bond features

    num_molecules1
        total number of molecules

    '''

    if not mol_shapes:
        mol_shapes = [t.shape for t in mol_tensors]

    num_molecules0, max_atoms0, num_atom_features = mol_shapes[0]
    num_molecules1, max_atoms1, max_degree1, num_bond_features = mol_shapes[1]
    num_molecules2, max_atoms2, max_degree2 = mol_shapes[2]

    num_molecules_vals = [num_molecules0, num_molecules1, num_molecules2]
    max_atoms_vals = [max_atoms0, max_atoms1, max_atoms2]
    max_degree_vals = [max_degree1, max_degree2]

    assert len(set(num_molecules_vals))==1, 'num_molecules does not match within tensors (found: {})'.format(num_molecules_vals)
    assert len(set(max_atoms_vals))==1, 'max_atoms does not match within tensors (found: {})'.format(max_atoms_vals)
    assert len(set(max_degree_vals))==1, 'max_degree does not match within tensors (found: {})'.format(max_degree_vals)

    return max_atoms1, max_degree1, num_atom_features, num_bond_features, num_molecules1


def is_iterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False


def zip_mixed(*mixed_iterables, **kwargs):
    ''' Zips a mix of iterables and non-iterables, non-iterables are repeated
    for each entry.

    # Arguments
        mixed_iterables (any type): unnamed arguments (just like `zip`)
        repeat_classes (list): named argument, which classes to repeat even though,
            they are in fact iterable

    '''

    repeat_classes = tuple(kwargs.get('repeat_classes', []))
    mixed_iterables = list(mixed_iterables)

    for i, item in enumerate(mixed_iterables):
        if not is_iterable(item):
            mixed_iterables[i] = cycle([item])

        if isinstance(item, repeat_classes):
            mixed_iterables[i] = cycle([item])

    return zip(*mixed_iterables)


def regression_metrics(y_true, y_predicted, nfeatures = None):
    """
    calculates metrics to evaluate regression models
    
    Parameters
    ----------
    y_true : list or 1D array
           actual values
    
    y_predicted : list or 1D array
                predicted values

    nfeatures : int, default = None
              number of features required to calculated adjusted R squared

    Returns
    -------
    metrics_df: dataframe with all metrics
    """
    metrics_dict = {}
    y_true = np.asarray(y_true)
    y_predicted = np.asarray(y_predicted)
    ndata = len(y_true)
    y_mean = np.mean(y_true)
    # actual errors
    e = y_true - y_predicted
    # relative errors
    re_flag = True
    if 0 in list(y_true):
        re_flag = False
    else:
        re = e/y_true
    # absolute errors
    ae = np.absolute(e)
    # squared errors
    se = np.square(e)

    metrics_dict['E'] = [list(e)]
    if re_flag == True:
        metrics_dict['RE'] = [list(re)]
    
    metrics_dict['AE'] = [list(ae)]
    metrics_dict['SE'] = [list(se)]

    var = np.mean(np.square(y_true - y_mean))
    
    metrics_dict['ME'] = np.mean(e)
    # mean absolute error
    mae = np.mean(ae)
    metrics_dict['MAE'] = mae
    
    # mean squared error
    mse = np.mean(se)
    metrics_dict['MSE'] = mse

    # root mean squared error
    rmse = np.sqrt(mse)
    metrics_dict['RMSE'] = rmse
    
    # mean squared log error
    if sum(y_true) == sum(np.abs(y_true)) and sum(y_predicted) == sum(np.abs(y_predicted)):
        msle = np.mean(np.square(np.log(1+y_true) - np.log(1+y_predicted)))
        metrics_dict['MSLE'] = msle
        rmsle = np.sqrt(msle)
        metrics_dict['RMSLE']=rmsle

    if re_flag == True:
        # mean absolute percentage error
        mape = np.mean(np.abs(re)) * 100
        metrics_dict['MAPE'] = mape
        # maximum absolute percentage error
        max_abs_perc_error = np.max(np.abs(re)) * 100
        metrics_dict['MaxAPE'] = max_abs_perc_error
        # root mean squared percentage error
        rmspe = np.sqrt(np.mean(np.square(re))) * 100
        metrics_dict['RMSPE'] = rmspe
        # mean percentage error
        mpe = np.mean(re) * 100
        metrics_dict['MPE'] = mpe
        
    # maximum absolute error
    max_ae = np.max(ae)
    metrics_dict['MaxAE'] = max_ae
    
    # difference between max error and min error
    delta_max_e = np.max(e) - np.min(e)
    metrics_dict['deltaMaxE'] = delta_max_e
    
    # R squared
    r2 = 1 - mse/var
    metrics_dict['r_squared'] = r2
    metrics_dict['std'] = np.sqrt(var)
    
    # adjusted R squared
    if nfeatures != None:
        adj_r2 = 1 - ((1-r2) * (ndata - 1)/(ndata - nfeatures -1))
        metrics_dict['adjusted_r_squared'] = adj_r2
    import pandas as pd
    metrics_df = pd.DataFrame.from_dict(metrics_dict)
    return metrics_df



def ConvertFile(file_path, from_format, to_format):
    """
    Convert a file from 'from_format' to 'to_format'
    using openbabel (https://openbabel.org/wiki/Babel).

    Parameters:
    ----------
    file_path : string or list
        string or list of strings containing paths of files that need to be converted

    from_format: string
        String of letters that specify the format of the file that needs to be converted.
        This will be checked against 'file_path' that is provided by the user.
        If the file_path does not contain 'from_format' an error message will be raised.
        List of possible 'from_format's are on https://openbabel.org/wiki/Babel

    to_format: string
        String of letters that specify the target file format or the desired format.
        An openbabel command is generated which converts the files specified by 'file_path' to the target format.
        List of possible 'to_format's are on https://openbabel.org/wiki/Babel

    Returns:
    ------
    converted_file_paths: str or list
        string or list of converted file paths depending on input type.

    Examples:
    --------
    >>> from chemml.datasets import load_xyz_polarizability
    >>> coordinates,polarizability = load_xyz_polarizability()
    >>> coordinates
    {1: {'file': 'chemml/datasets/data/organic_xyz/1_opt.xyz', ...
    >>> from chemml.utils import ConvertFile
    >>> converted_file_paths = ConvertFile(file_path=coordinates,from_format='xyz',to_format='cml')
    {1: {'file': 'chemml/datasets/data/organic_xyz/1_opt.cml'}, 2: ...

    """
    

    if isinstance(file_path, str):
        if not from_format == file_path[-len(from_format):]:
            msg = 'file format is not the same as from_format'
            raise ValueError(msg)
        elif os.path.exists(file_path) == False:
            msg = 'file does not exist at '+ file_path
            raise FileNotFoundError(msg)
        else:
            ob_from_format = '-i' + from_format
            ob_to_format = '-o' + to_format
            path = file_path[:file_path.rfind('.') + 1] # = path-to-file.
            command = 'obabel ' + ob_from_format + ' ' + file_path + ' ' + ob_to_format + ' -O ' + path + to_format 
            # path + selt.to_format = path-to-file.newformat
            # print(command)
            os.system(command)
            converted_file_paths = path + to_format


    elif isinstance(file_path, list):
        converted_file_paths = []
        
        # for it in range(1, len(self.file_path) + 1):
        for fpath in file_path:
            # fpath = self.file_path[it]['file']
            if not fpath[-len(from_format):] == from_format:
                msg = 'file format is not the same as from_format'
                raise ValueError(msg)
            elif os.path.exists(fpath) == False:
                msg = 'file does not exist at '+ fpath
                raise FileNotFoundError(msg)
            else:
                ob_from_format = '-i' + from_format
                ob_to_format = '-o' + to_format
                path = fpath[:fpath.rfind('.') + 1]
                command = 'obabel ' + ob_from_format + ' ' + fpath + ' ' + ob_to_format + ' -O ' + path + to_format
                # print(command)
                os.system(command)
                converted_file_paths.append(path + to_format)

    else:
        converted_file_paths = None
        raise ValueError('File path must be a string or a list of strings.')
    
    return converted_file_paths
    







