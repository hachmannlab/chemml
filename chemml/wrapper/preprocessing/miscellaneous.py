"""
miscellaneous preprocessing methods just to make ChemML Wrapper more accessible to users.
"""
import numpy as np
import pandas as pd
import warnings
import os

from chemml.utils import std_datetime_str


class SplitColumns(object):
    """
    This method splits a pandas dataframe by columns.

    Parameters
    ----------
    selection : int, optional (default = 1)
        if positive integer, it's the number of columns to be selected from left side of dataframe and returns as first data frame (df1).
        if negative integer, it's the number of columns to be selected from right side of dataframe and returns as first data frame (df1).
    """
    def __init__(self, selection=1):
        self.selection = selection

    def fit(self, X):
        """
        The main function to split the input dataframe.

        Parameters
        ----------
        X: array
            the input array

        Returns
        -------
        X1 : array
            The array resulted based on the selection parameter

        X2 : array
            The array of columns that are not selected
        """
        X = np.array(X)
        if not isinstance(X, np.ndarray):
            msg = 'The input `X` must be an array-like data structure.'
            raise ValueError(msg)

        # split by int
        if isinstance(self.selection, int):
            if self.selection >= X.shape[1]:
                msg = 'Selected all columns, the second output data frame is None.'
                warnings.warn(msg)
            if self.selection > 0 :
                X1 = X[:, :self.selection]
                X2 = X[:, self.selection:]
            elif self.selection<0:
                X1 = X[:, self.selection:]
                X2 = X[:, :self.selection]
            else: #but why??
                X1 = None
                X2 = X
        else:
            msg = "The input `selection` must ba an integer"
            raise ValueError(msg)
        return X1, X2


class SaveCSV(object):
    """
    Write pandas DataFrame to a comma-seprated-values(CSV) file.

    Parameters
    ----------
    file_path : str
        The path for the CSV file

    record_time : bool, optional(default=False)
        If True, the current time will be added to the file name.

    index : bool, optional(default=False)
        If True, the index of the dataframe will be also stored as the first column.

    header : bool, optional(default=False)
        If True, the header of the dataframe will be stored.

    """
    def __init__(self, file_path, record_time = False,
                 index = False, header = True):
        self.filename = os.path.basename(file_path)
        if len(self.filename.strip())==0:
            msg = "The input `file_path` must contain a file name."
            raise ValueError(msg)
        self.output_directory = os.path.dirname(file_path)
        self.record_time = record_time
        self.index = index
        self.header = header

    def fit(self, df, main_directory=''):
        """
        Write DataFrame to a comma-seprated-values CSV) file.

        Parameters
        ----------
        df : pandas DataFrame
            The input pandas dataframe

        main_directory : str, optional (default='')
            if there is a main directory for entire chemml wrapper project
        """
        if not isinstance(df, pd.DataFrame):
            msg = 'The input `df` must be a pandas dataframe.'
            raise TypeError(msg)

        # create final output dir
        self.output_directory = os.path.join(main_directory, self.output_directory)
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)

        # record time
        if self.record_time:
            self.file_path = '%s/%s_%s'%(self.output_directory, self.filename,std_datetime_str())
        else:
            self.file_path = '%s/%s' % (self.output_directory, self.filename)


        # store csv file
        df.to_csv(self.file_path, index=self.index, header=self.header)


class SaveFile(object):
    """
    Write any input data to a file in string format.
    This is good for making text files and keeping track of metadata.

    Parameters
    ----------
    file_path : str
        The path for the CSV file

    record_time : bool, optional(default=False)
        If True, the current time will be added to the file name.

    """

    def __init__(self, file_path, record_time = False):
        self.filename = os.path.basename(file_path)
        if len(self.filename.strip())==0:
            msg = "The input `file_path` must contain a file name."
            raise ValueError(msg)
        self.output_directory = os.path.dirname(file_path)
        self.record_time = record_time

    def fit(self, X, main_directory=''):
        """
        This function Write an input data X to a file as a string.

        Parameters
        ----------
        df : pandas DataFrame
            The input pandas dataframe

        main_directory : str, optional (default='')
            if there is a main directory for entire chemml wrapper project
        """

        # create final output dir
        self.output_directory = os.path.join(main_directory, self.output_directory)
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)

        if self.record_time:
            self.file_path = '%s/%s_%s'%(self.output_directory, self.filename,std_datetime_str())
        else:
            self.file_path = '%s/%s' % (self.output_directory, self.filename)

        # store file
        with open(self.file_path, 'a') as file:
            file.write('%s\n' % str(X))
