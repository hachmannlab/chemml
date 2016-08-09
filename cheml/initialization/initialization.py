import os
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
        if integer, shows number of columns from the first of data frame to be cut as first data frame (X1)
        if list, is array of labels to be cut as first data frame (X1)

    :return: two pandas data frame: X1 and X2
    """
    if not isinstance(X,pd.DataFrame):
        msg = 'X must be a pandas dataframe'
        raise TypeError(msg)
    if isinstance(select,list):
        X1 = X.loc[:,select]
        X2 = X.drop(select,axis=1)
    elif isinstance(select,int):
        if select >= X.shape[1]:
            msg = 'The first output data frame is empty, because passed a bigger number than actual number of columns'
            warnings.warn(msg)
        X1 = X.iloc[:,:select]
        X2 = X.iloc[:,select:]
    else:
        msg = "parameter 'select' must ba a list or an integer"
        raise TypeError(msg)
    return X1, X2

class SaveFile(object):
    """
    Write DataFrame to a comma-seprated values(csv) file
    :param output_directory: string, the output directory to save output files

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
        :param X: pandas DataFrame
        :param main_directory: string, if there is any main directory for entire cheml project
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

