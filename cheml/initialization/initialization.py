import os
import copy
import shutil
import pandas as pd
import warnings

from ..utils.utilities import std_datetime_str

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
    if not isinstance(X1,pd.DataFrame) or not isinstance(X2,pd.DataFrame):
        msg = 'both X1 and X2 must be pandas dataframe'
        raise TypeError(msg)
    if X1.shape[0] != X2.shape[0]:
        msg= 'Two input data frames should be in the same length'
        raise ValueError(msg)
    X = X1.join(X2,lsuffix='_X1',rsuffix='_X2')
    return X

def Split(X,select=1):
    """
    split data frame by columns

    :param X: pandas data frame
        original pandas data frame

    :param select: integer or list (default = 1)
        if integer, shows number of columns from the end of data frame to be cut as second data frame
        if list, is array of labels

    :return: two pandas data frame: X1 and X2
    """
    if not isinstance(X,pd.DataFrame):
        msg = 'X must be a pandas dataframe'
        raise TypeError(msg)
    if isinstance(select,list):
        X2 = X.loc[:,select]
        X1 = X.drop(select,axis=1)
    elif isinstance(select,int):
        if select >= X.shape[1]:
            msg = 'The first output data frame is empty, because passed a bigger number than actual number of columns'
            warnings.warn(msg)
        X2 = X.iloc[:,-select:]
        X1 = X.iloc[:,:-select]
    else:
        msg = "parameter 'select' can only ba a list or integer"
        raise TypeError(msg)
    return X1, X2

class Settings(object):
    """
    makes the output directory.

    Parameters
    ----------
    output_directory: String, (default = "CheML.out")
        The directory path/name to store all the results and outputs

    input_copy: Boolean, (default = True)
        If True, keeps a copy of input script in the output_directory

    Returns
    -------
    output_directory
    """
    def __init__(self,output_directory="CMLWrapper.out", InputScript_copy = True):
        self.output_directory = output_directory
        self.InputScript_copy = InputScript_copy

    def fit(self,InputScript):
        initial_output_dir = copy.deepcopy(output_directory)
        i = 0
        while os.path.exists(output_directory):
            i+=1
            output_directory = initial_output_dir + '%i'%i
        os.makedirs(output_directory)
        # log_file = open(output_directory+'/'+logfile,'a',0)
        # error_file = open(output_directory+'/'+errorfile,'a',0)
        if self.InputScript_copy:
            shutil.copyfile(InputScript, output_directory + '/InputScript.txt')
        return output_directory

class SaveFile(object):
    """
    Write DataFrame to a comma-seprated values(csv) file
    """
    def __init__(self, filename):
        self.filename = filename

    def fit(X,output_directory):
        """
        Write DataFrame to a comma-seprated values(csv) file
        :param X: pandas DataFrame
        :param output_directory: string, the output directory to save output files
        :return: nothing
        """
        X.to_csv('%s/%s_%s.csv'%(output_directory,self.filename,std_datetime_str()), index=False)
