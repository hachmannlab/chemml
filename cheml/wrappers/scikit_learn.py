from .sct_utils import isfloat, islist, istuple, isnpdot, std_datetime_str

def Imputer_dataframe(transformer, df):
    """ keep track of features (columns) that can be removed or changed in the
        Imputer by transforming data back to pandas dataframe structure. This happens based on
        the "statistics_" attribute of Imputer.

    Parameters
    ----------
    transformer: sklearn Imputer class
         The class with adjusted parameters.

    df: Pandas dataframe
        The dataframe that imputer is going to deal with.

    Returns
    -------
    transformed data frame
    fitted imputer class
    """
    df_columns = list(df.columns)
    df = transformer.fit_transform(df)
    if df.shape[1] == 0:
        warnings.warn("empty dataframe: all columns have been removed",Warning)
        return transformer, df
    else:
        stats = transformer.statistics_
        nan_ind = [i for i,val in enumerate(stats) if np.isnan(val)]
        df_columns = list_del_indices(df_columns, nan_ind)
        df = pd.DataFrame(df,columns=df_columns)
        return df

def transformer_dataframe(transformer, df):
    """ keep track of features (columns) that can be removed or changed in the
        Scaler by transforming data back to pandas dataframe structure.

    Parameters
    ----------
    scaler: sklearn Scaler class
        The class with adjusted parameters.

    df: Pandas dataframe
        The dataframe that scaler is going to deal with.

    Returns
    -------
    transformed data frame
    fitted scaler class

    """
    df_columns = list(df.columns)
    df = transformer.fit_transform(df)
    if df.shape[1] == 0:
        warnings.warn("empty dataframe: all columns have been removed",Warning)
    if df.shape[1] == len(df_columns):
        df = pd.DataFrame(df,columns=df_columns)
    else:
        warnings.warn("number of columns before and after transform doesn't match",Warning)
    return df

def selector_dataframe(transformer, df, tf):
    """ keep track of features (columns) that can be removed or changed in the
        VarianceThreshold by transforming data back to pandas dataframe structure.
        This happens based on the "get_support" method of selector.

    Parameters
    ----------
    imputer: sklearn VarianceThreshold class
         The class with adjusted parameters.

    df: Pandas dataframe
        data frame

    tf: Pandas dataframe
        target frame

    Returns
    -------
    transformed data frame
    fitted imputer class
    """
    df_columns = list(df.columns)
    df = transformer.fit_transform(df,tf)
    if df.shape[1] == 0:
        warnings.warn("empty dataframe: all columns have been removed",Warning)
        return transformer, df
    else:
        retained_features_ind = sel.get_support(True)
        df_columns = [df_columns[i] for i in retained_features_ind]
        df = pd.DataFrame(df,columns=df_columns)
        return df

class StandardScaler(object):
    def __init__(self, Base,parameters,iblock):
        self.Base = Base
        self.parameters = parameters
        self.iblock = iblock
        self.fit()
        self.receive()
        self.transform_predict()

    def receive(self):
        legal_inputs = ['X','Y']
        self.X = None
        self.Y = None
        receive = [node for node in self.Base.graph if node[1]==self.iblock]
        for node in receive:
            if node[2] in legal_inputs:
                ind = legal_inputs.index(node[2])
                count[ind] += 1
                if count[ind]>1:
                    msg = 'only one input per each available legal input can be accepted.\
                           list of legal inputs in function #%i: %s'%(str(legal_inputs),self.iblock+1)
                    raise IOError(msg)
            else:
                msg = 'received an unvalid input in function #%i, sent by function #%i' %(self.iblock+1,node[0]+1)
                raise IOError(msg)
            if


    def fit(self):
        from sklearn.preprocessing import StandardScaler
        self.model = StandardScaler(**self.parameters)

    def transform_predict(self):
        self.model.



class Sklearn_Base(object):
    def __init__(self, Base, SuperFunction,function,parameters):
        self.Base = Base
        self.SuperFunction = SuperFunction
        self.function = function
        self.parameters = parameters

    # def call(self):
    #     interface = exec(self.function)

