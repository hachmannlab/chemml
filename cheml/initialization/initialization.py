import os
import pandas as pd
import warnings

from ..utils.utilities import std_datetime_str

def ReadTable(filepath, header=None, skipcolumns=0, skiprows=0):
    """
    Read general delimited file into DataFrame.

    Parameters:
    ----------
    filepath: string
        The path/name of input file in any format
        Note that engine may switch from 'c' to python for some of the delimiters (pandas.read_table)

    header: None or 0 (default = None)
        0 if the label of columns is available in the file

    skipcolumns: integer (default = 0)
        number of columns to skip at the start of the file

    skiprows: integer (default = 0)
        number of rows to skip at the start of the file


    Return:
    -------
    pandas data frame

    """
    X = pd.read_table(filepath, sep=None, skiprows=skiprows, header=header, engine='python')
    if skipcolumns>0:
        X = X.drop(X.columns[0:skipcolumns], 1)
        if header==None:
            X.columns = range(len(X.columns))
    return X

def Merge(X1, X2):
    """
    todo: add more functionality for header overlaps
    Merge same length data frames.

    :param X1: pandas data frame
        first data frame
    :param X2: pandas data frame
        second data frame
    :return: pandas data frame
    """
    if not isinstance(X1,pd.DataFrame) or not isinstance(X2,pd.DataFrame):
        msg = 'both X1 and X2 must be pandas dataframe'
        raise TypeError(msg)
    if X1.shape[0] != X2.shape[0]:
        msg= 'Two input data frames should be in the same length'
        raise ValueError(msg)
    X = X1.join(X2,lsuffix='_X1',rsuffix='_X2')
    return X

class Split(object):
    """
    split data frame by columns

    :param select: integer or list (default = 1)
        if integer, shows number of columns from the first of data frame to be cut as first data frame (X1)
        if list, is array of labels to be cut as first data frame (X1)
    :return: two pandas data frame: X1 and X2
    """
    def __init__(self,selection=1):
        self.selection = selection

    def fit(self,X):
        """
        fit the split task to the input data frame

        :param X:  pandas data frame
        original pandas data frame
        :return: two pandas data frame: X1 and X2
        """
        if not isinstance(X,pd.DataFrame):
            msg = 'X must be a pandas dataframe'
            raise TypeError(msg)
        if isinstance(self.selection,list):
            X1 = X.loc[:,self.selection]
            X2 = X.drop(self.selection,axis=1)
        elif isinstance(self.selection,int):
            if self.selection >= X.shape[1]:
                msg = 'The first output data frame is empty, because passed a bigger number than actual number of columns'
                warnings.warn(msg)
            X1 = X.iloc[:,:self.selection]
            X2 = X.iloc[:,self.selection:]
        else:
            msg = "selection parameter must ba a list or an integer"
            raise TypeError(msg)
        return X1, X2

# class Match(object):
#     def __init__(self,):
#
#     def fit (X1)
#     def transform(self,X1,X2,header):
#         df = pd.concat([X1, X2], axis=1, join='inner')

class SaveFile(object):
    """
    Write DataFrame to a comma-seprated values(csv) file
    :param: output_directory: string, the output directory to save output files

    """
    def __init__(self, filename, output_directory = None, record_time = False, format ='csv',
                 index = False, header = True):
        self.filename = filename
        self.record_time = record_time
        self.output_directory = output_directory
        self.format = format
        self.index = index
        self.header = header

    def fit(self, X, main_directory=None):
        """
        Write DataFrame to a comma-seprated values (csv) file
        :param: X: pandas DataFrame
        :param: main_directory: string, if there is any main directory for entire cheml project
        :return: nothing
        """
        if not isinstance(X, pd.DataFrame):
            msg = 'X must be a pandas dataframe'
            raise TypeError(msg)
        if main_directory:
            self.output_directory = main_directory + '/' + self.output_directory
        if self.output_directory:
            if not os.path.exists(self.output_directory):
                os.makedirs(self.output_directory)
            if self.record_time:
                self.file_path = '%s/%s_%s.%s'%(self.output_directory, self.filename,std_datetime_str(),self.format)
                X.to_csv(self.file_path, index=self.index, header = self.header)
            else:
                self.file_path = '%s/%s.%s' % (self.output_directory, self.filename,self.format)
                X.to_csv(self.file_path, index=self.index, header = self.header)
        else:
            if self.record_time:
                self.file_path = '%s_%s.%s'%(self.filename,std_datetime_str(),self.format)
                X.to_csv(self.file_path, index=self.index, header = self.header)
            else:
                self.file_path = '%s.%s' %(self.filename,self.format)
                X.to_csv(self.file_path, index=self.index, header = self.header)


def slurm_script(block):
    """(slurm_script):
        if part of your code must be run on a cluster and you need to make a slurm
        script for that purpose, this function helps you to do so.

    Parameters
    ----------
    style: string, optional(default=exclusive)
        Available options:
            - exclusive : makes the slurm script based on exclusive selection of cores per nodes.

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
    style = block['parameters']['style'][1:-1]
    pyscript_file = cmlnb["file_name"]
    nnodes = int(block['parameters']['nnodes'])
    input_slurm_script = block['parameters']['input_slurm_script'][1:-1]
    output_slurm_script = block['parameters']['output_slurm_script'][1:-1]

    cmlnb["run"] = "# how to run: sbatch %s" % output_slurm_script

    if style == 'exclusive':
        if input_slurm_script != 'None':
            file = ['#!/bin/sh\n', '#SBATCH --time=99:00:00\n', '#SBATCH --job-name="nn"\n',
                    '#SBATCH --output=nn.out\n', '#SBATCH --clusters=chemistry\n', '#SBATCH --partition=beta\n',
                    '#SBATCH --account=pi-hachmann\n', '#SBATCH --exclusive\n', '#SBATCH --nodes=1\n', '\n',
                    '# ====================================================\n', '# For 16-core nodes\n',
                    '# ====================================================\n', '#SBATCH --constraint=CPU-E5-2630v3\n',
                    '#SBATCH --tasks-per-node=1\n', '#SBATCH --mem=64000\n', '\n', '\n',
                    'echo "SLURM job ID         = "$SLURM_JOB_ID\n',
                    'echo "Working Dir          = "$SLURM_SUBMIT_DIR\n', 'echo "Temporary scratch    = "$SLURMTMPDIR\n',
                    'echo "Compute Nodes        = "$SLURM_NODELIST\n', 'echo "Number of Processors = "$SLURM_NPROCS\n',
                    'echo "Number of Nodes      = "$SLURM_NNODES\n', 'echo "Tasks per Node       = "$TPN\n',
                    'echo "Memory per Node      = "$SLURM_MEM_PER_NODE\n', '\n', 'ulimit -s unlimited\n',
                    'module load intel-mpi\n', 'module load python\n', 'module list\n',
                    'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/projects/hachmann/packages/Anaconda:/projects/hachmann/packages/rdkit-Release_2015_03_1:/user/m27/pkg/openbabel/2.3.2/lib\n',
                    'date\n', '\n', '\n', 'echo "Launch job"\n', 'export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so\n',
                    'export I_MPI_FABRICS=shm:tcp\n', '\n', 'mpirun -np 2 python test.py\n']
            file[8] = '#SBATCH --nodes=%i\n' % nnodes
            file[-1] = 'mpirun -np %i python %s\n' % (nnodes, pyscript_file)
        else:
            file = open(input_slurm_script, 'r')
            file = file.readlines()
            exclusive_flag = False
            nodes_flag = False
            np_flag = False
            for i, line in enumerate(file):
                if '--exclusive' in line:
                    exclusive_flag = True
                elif '--nodes' in line:
                    nodes_flag = True
                    ind = line.index('--nodes')
                    file[i] = line[:ind] + '--nodes=%i\n' % nnodes
                elif '-np' in line:
                    np_flag = True
                    ind = line.index('--nodes')
                    file[i] = line[:ind] + '--nodes=%i\n' % nnodes
            if not exclusive_flag:
                file = file[0] + ['#SBATCH --exclusive\n'] + file[1:]
                msg = "The --exclusive option is not available in the slurm script. We added '#SBATCH --exclusive' to the first of file."
                warnings.warn(msg, UserWarning)
            if not nodes_flag:
                file = file[0] + ['#SBATCH --nodes=%i\n' % nnodes] + file[1:]
                msg = "The --nodes option is not available in the slurm script. We added '#SBATCH --nodes=%i' to the first of file." % nnodes
                warnings.warn(msg, UserWarning)
            if not np_flag:
                file.append('mpirun -np %i python %s\n' % (nnodes, pyscript_file))
                msg = "The -np option is not available in the slurm script. We added 'mpirun -np %i python %s'to the end of file." % (
                nnodes, pyscript_file)
                warnings.warn(msg, UserWarning)

        script = open(output_slurm_script, 'w')
        for line in file:
            script.write(line)
        script.close()