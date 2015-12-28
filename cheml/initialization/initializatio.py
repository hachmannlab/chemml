import os
import copy
import shutil
import pandas as pd


__all__ = [
    'input',
    'output',
]
###################################################################################################
# TODO: 
# - copy pyscript into output_directory and remove original version
###################################################################################################

def input(data_path, data_label, n_skip_columns=0, n_skip_rows=0, target_path=None, target_label=None):
    """
    Read input data files. 
    
    Parameters
    ----------
    data_path: the path/name of input data file in any format
               engine can switch from 'c' to python for some of the delimiters (pandas.read_csv)
    data_label: label of the columns are available in the file or not
    n_skip_columns: number of columns to skip at the start of the file  
    n_skip_rows: number of rows to skip at the start of the file
    target_path: the path/name of target file in the csv format
    target_label: label of the target columns
    
    Returns
    -------
    data = input data in the pandas dataframe format
    target = column of target in the pandas dataframe format
    """   
    if data_label:
        data = pd.read_csv(data_path, sep=None, skiprows=n_skip_rows)
        data = data.drop(data.columns[0:n_skip_columns],1)
    else:
        data = pd.read_csv(data_path, sep=None, skiprows=n_skip_rows, header=None)
        data = data.drop(data.columns[0:n_skip_columns],1)
        data.columns = range(len(data.columns))
    
    if target_label:
        target = pd.read_csv(target_path)
    else:
        target = pd.read_csv(target_path, header=None)
    return data, target

def output(output_directory="ChemML.results" ,logfile="log.txt",errorfile="error.txt"):
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
    return output_directory, logfile, errorfile, tmp_folder

