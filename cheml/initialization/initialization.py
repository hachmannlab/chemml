import os
import copy
import shutil
import pandas as pd


__all__ = [
    'File',
    'output',
]

def File(filepath, header=None, skipcolumns=0, skiprows=0):
    """
    Read input data file.

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
    X = pd.read_table(filepath, sep=None, skiprows=skiprows, header=header)
    if skipcolumns>0:
        X = X.drop(X.columns[0:skipcolumns], 1)
        if header==None:
            X.columns = range(len(X.columns))
    return X

def Merge(X1, X2):
    """
    todo: add more functionality for header overlaps
    Merge same length data frames.

    :param X_1: pandas data frame
        first data frame
    :param X_2: pandas data frame
        second data frame
    :return: pandas data frame
    """
    X = X1.join(X2,lsuffix='_X1',rsuffix='_X2')
    return X

def output(output_directory="CheML.out" ,logfile="log.txt",errorfile="error.txt"):
    """
    make all the output files and folders in the output directory. 
    These out put files keep track of all the changes in the rest of program.
    
    Parameters
    ----------
    output_directory: The directory path/name to store all the results and outputs 
    logfile: The name of log file
    errorfile: The name of error file
    
    Returns
    -------
    output_directory: The directory path/name after checking the existence of initial path
    logfile: log file in the output_directory
    errorfile: error file in the output_directory
    tmp_folder: temporary folder in the output_directory, will be removed at the end.
    """
    initial_output_dir = copy.deepcopy(output_directory)
    i = 0
    while os.path.exists(output_directory):
        i+=1
        output_directory = initial_output_dir + '%i'%i
    os.makedirs(output_directory)
    log_file = open(output_directory+'/'+logfile,'a',0)
    error_file = open(output_directory+'/'+errorfile,'a',0)
    tmp_folder = output_directory +'/temporary'
    os.makedirs(tmp_folder)
    return output_directory, logfile, errorfile

