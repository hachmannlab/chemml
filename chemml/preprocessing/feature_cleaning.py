import numpy as np
import pandas as pd


def MissingValues(df, strategy="ignore_row",
                  string_as_null=True,
                  inf_as_null=True,
                  missing_values=None):
    """
    find missing values and interpolate/replace or remove them.

    Parameters
    ----------
    df : pandas dataframe

    strategy: string, optional (default="ignore_row")

        list of strategies:
        - interpolate: interpolate based on sorted target values
        - zero: set to the zero
        - ignore_row: remove the entire row in data and target
        - ignore_column: remove the entire column in data and target

    string_as_null: boolean, optional (default=True)
        If True non numeric elements are considered to be null in computations.

    missing_values: list, optional (default=None)
        where you define specific formats of missing values. It is a list of string, float or integer values.

    inf_as_null: boolean, optional (default=True)
        If True inf and -inf elements are considered to be null in computations.

    Returns
    -------
    dataframe

    Notes
    ----------
    mask is a binary vector whose length is the number of rows/indices in the df. The index of each bit shows
    if the row/column in the same position has been removed or not.
    The goal is keeping track of removed rows/columns to change the target data frame or other input data frames based
    on that. The mask can later be used in the transform method to change other data frames in the same way.
    """
    from chemml.utils import check_object_col
    if inf_as_null == True:
        df.replace([np.inf, -np.inf, 'inf', '-inf'], np.nan, inplace=True)
    if string_as_null == True:
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if isinstance(missing_values, (list, tuple)):
        for pattern in missing_values:
            df.replace(pattern, np.nan, True)

    df = check_object_col(df, 'df')
    # drop null columns
    df = df.dropna(axis=1, how='all', inplace=False)

    if strategy == 'zero':
        df = df.fillna(0)
        return df
    elif strategy == 'ignore_row':
        dfi = df.index
        df = df.dropna(axis=0, how='any', inplace=False)
        mask = [i in df.index for i in dfi]
        mask = pd.Series(mask, index=dfi)
        # mask = pd.notnull(df).all(1)
        # df = df[mask]
        return df
    elif strategy == 'ignore_column':
        dfc = df.columns
        df = df.dropna(axis=1, how='any', inplace=False)
        mask = [i in df.columns for i in dfc]
        mask = pd.Series(mask, index=dfc)
        # mask = pd.notnull(df).all(0)
        # df = df.T[mask].T
        return df
    elif strategy == 'interpolate':
        df = df.interpolate()
        df = df.fillna(
            method='ffill', axis=1, inplace=False
        )  # because of nan in the first and last element of column
        return df
    else:
        msg = "Wrong strategy has been passed"
        raise TypeError(msg)


def Outliers(df, m=2.0, strategy='median'):
    """
    remove all rows where the values of a certain column are within an specified
    standard deviation from mean/median.

    Parameters
    ----------
    df: pandas dataframe
        input dataframe

    m: float, optional (default=3.0)
        the outlier threshold with respect to the standard deviation

    strategy: string, optional (default='median')
        available options: 'mean' and 'median'
        Values of each column will be compared to the 'mean' or 'median' of that column.

    Returns
    -------
    dataframe

    Notes
    -----
    We highly recommend you to remove constant columns first and then remove outliers.

    """
    if strategy == 'mean':
        mask = ((df - df.mean()).abs() <= m * df.std(ddof=0)).T.all()
    elif strategy == 'median':
        mask = (((df - df.median()).abs()) <=
                m * df.std(ddof=0)).T.all()
    df = df.loc[mask, :]
    removed_rows_ = np.array(mask[mask == False].index)
    return df


def ConstantColumns(df):
    """
    Removes single-value columns.
    NOTE: Deprecated in v1.3.4, now backend uses RemoveInvFeatures. Will be removed in v1.4.

    Parameters
    ----------
    df: pandas dataframe
        input dataframe

    Returns
    -------
    df: pandas dataframe

    """
    return RemoveInvFeatures(df,
                             filter_single_value=True,
                             sanitize_binary=False,
                             sanitize_nonbinary=False,
                             use_variance_filtering=False,
                             keep_filtered_columns=False)


def RemoveCorrFeatures(df, correlation_threshold=0.9):
    """
    remove highly correlated feature columns.

    Parameters
    ----------
    df: pandas dataframe
        input dataframe

    correlation_threshold: float, optional (default=0.9)
        absolute correlation threshold. If any pair of columns has absolute
        correlation larger than this threshold, the later column (based on
        column order) is removed.

    Returns
    -------
    df: pandas dataframe
        dataframe with highly correlated columns removed
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("'df' must be a pandas DataFrame.")
    if not isinstance(correlation_threshold, (int, float)):
        raise TypeError("'correlation_threshold' must be numeric.")
    if correlation_threshold <= 0 or correlation_threshold > 1:
        raise ValueError("'correlation_threshold' must be in the interval (0, 1].")

    clean_df = df.copy()
    numeric_df = clean_df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return clean_df

    corr_matrix = numeric_df.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > correlation_threshold)]

    if len(to_drop) == 0:
        return clean_df
    return clean_df.drop(columns=to_drop)


def RemoveInvFeatures(df,
                      sanitize_threshold=0.9,
                      filter_single_value=True,
                      sanitize_binary=True,
                      sanitize_nonbinary=True,
                      use_variance_filtering=True,
                      variance_threshold=0.01,
                      keep_filtered_columns=False):
    """
    remove invariant or near-invariant feature columns.

    Parameters
    ----------
    df: pandas dataframe
        input dataframe

    sanitize_threshold: float, optional (default=0.9)
        dominant-frequency threshold used for invariant filtering. Columns
        with dominant value frequency greater than or equal to this fraction
        of rows are removed.

    sanitize_nonbinary: boolean, optional (default=True)
        if True, remove nonbinary columns (nunique > 2) where one value is
        dominant in more than 90 percent of rows.

    use_variance_filtering: boolean, optional (default=True)
        if True, remove numeric columns with variance less than
        ``variance_threshold``.

    variance_threshold: float, optional (default=0.01)
        minimum variance required to keep a numeric column.

    keep_filtered_columns: boolean, optional (default=False)
        if True, returns a tuple ``(clean_df, removed_df)`` where ``removed_df``
        contains all removed columns from the original input dataframe.

    Returns
    -------
    df: pandas dataframe
        cleaned dataframe when ``keep_filtered_columns=False``

    tuple
        ``(clean_df, removed_df)`` when ``keep_filtered_columns=True``
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("'df' must be a pandas DataFrame.")
    if not isinstance(filter_single_value, bool):
        raise TypeError("'filter_single_value' must be a boolean.")
    if not isinstance(sanitize_binary, bool):
        raise TypeError("'sanitize_binary' must be a boolean.")
    if not isinstance(sanitize_nonbinary, bool):
        raise TypeError("'sanitize_nonbinary' must be a boolean.")
    if not isinstance(sanitize_threshold, (int, float)):
        raise TypeError("'sanitize_threshold' must be numeric.")
    if sanitize_threshold <= 0 or sanitize_threshold > 1:
        raise ValueError("'sanitize_threshold' must be in the interval (0, 1].")
    if not isinstance(use_variance_filtering, bool):
        raise TypeError("'use_variance_filtering' must be a boolean.")
    if not isinstance(keep_filtered_columns, bool):
        raise TypeError("'keep_filtered_columns' must be a boolean.")
    if not isinstance(variance_threshold, (int, float)):
        raise TypeError("'variance_threshold' must be numeric.")
    if variance_threshold < 0:
        raise ValueError("'variance_threshold' must be non-negative.")

    clean_df = df.copy()
    nunique_counts = {col: clean_df[col].nunique(dropna=False) for col in clean_df.columns}
    single_value_cols = []
    invariant_binary = []
    removed_columns = []

    # Remove single-value columns if requested.
    if filter_single_value:
        single_value_cols = [col for col in clean_df.columns if nunique_counts[col] == 1]
        if len(single_value_cols) > 0:
            clean_df = clean_df.drop(columns=single_value_cols)
            removed_columns.extend(single_value_cols)

    # Remove binary columns dominated by one value and all single-value columns.
    if sanitize_binary:
        binary_cols = [col for col in clean_df.columns if nunique_counts[col] == 2]
        binary_df = clean_df[binary_cols]
        invariant_binary = [
            col for col in binary_df.columns
            if binary_df[col].value_counts(dropna=False).max() >= sanitize_threshold * len(binary_df)
        ]
        if len(invariant_binary) > 0:
            clean_df = clean_df.drop(columns=invariant_binary)
            removed_columns.extend(invariant_binary)
    
    # Remove nonbinary columns dominated by one value.
    if sanitize_nonbinary:
        invariant_nonbinary = [
            col for col in clean_df.columns
            if nunique_counts[col] > 2
            and clean_df[col].value_counts(dropna=False).max() >= sanitize_threshold * len(clean_df)
        ]
        if len(invariant_nonbinary) > 0:
            clean_df = clean_df.drop(columns=invariant_nonbinary)
            removed_columns.extend(invariant_nonbinary)

    # Remove low-variance numeric columns.
    if use_variance_filtering:
        numeric_df = clean_df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] > 0:
            variances = numeric_df.var()
            low_variance_cols = variances[variances < variance_threshold].index.tolist()
            if len(low_variance_cols) > 0:
                clean_df = clean_df.drop(columns=low_variance_cols)
                removed_columns.extend(low_variance_cols)
    
    # Keep removed columns in original order and without duplicates.
    removed_unique = []
    for col in df.columns:
        if col in removed_columns and col not in removed_unique:
            removed_unique.append(col)

    if keep_filtered_columns:
        removed_df = df[removed_unique].copy()
        return clean_df, removed_df
    return clean_df
