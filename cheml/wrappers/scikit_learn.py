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

def _check_parameters(fn, parameters):
    """todo: finish
    """
    pass

class StandardScaler(object):
    def __init__(self, Base,parameters,iblock):
        self.Base = Base
        self.parameters = parameters
        self.iblock = iblock
        self.legal_inputs = {'X':None,'Y':None}
        self.legal_outputs = ['StandardScaler_api':None, 'X':None, 'Y':None]
        self.receive()
        self.fit()
        self.transform_predict()
        self.send(self)

    def receive(self):
        recv = [edge for edge in self.Base.graph if edge[2]==self.iblock]
        self.Base.graph = [edge for edge in self.Base.graph if edge[2]!=self.iblock]
        # check received tokens
        count = [0] * len(self.legal_inputs)
        for edge in recv:
            if edge[3] in self.legal_inputs:
                ind = self.legal_inputs.index(edge[2])
                count[ind] += 1
                if count[ind]>1:
                    msg = 'only one input per each available input can be received.\
                           list of legal inputs in function #%i: %s'%(str(self.legal_inputs),self.iblock+1)
                    raise IOError(msg)
            else:
                msg = "received a non valid input token '%s' in function #%i, sent by function #%i" %(edge[3],self.iblock+1,edge[0]+1)
                raise IOError(msg)
        for edge in recv:
            key = edge[0:2]
            if key in self.Base.send:
                value = self.Base.send[key][0]
                self.legal_inputs[edge[3]] = value
                self.Base.send[key][1] -= 1
                if self.Base.send[key][1] == 0:
                    del self.Base.send[key]
            else:
                msg = 'broken pipe in edge %s - nothing has been sent'%str(edge)
                raise IOError(msg)

    def fit(self):
        from sklearn.preprocessing import StandardScaler
        self.model = StandardScaler(**self.parameters)

    def transform_predict(self):
        self.model.

    def send(self):
        send = [edge for edge in self.Base.graph if edge[0]==self.iblock]
        for edge in send:
            key = edge[0:1]
            token = edge[1]
            if token not in self.legal_outputs:
                msg = "asked to send a non valid output token '%s' in function #%i" % (token, self.iblock + 1)
                raise NameError(msg)
            if key in self.Base.send:
                self.Base.send[key][1] += 1
            else:
                self.Base.send[key] = [self.legal_outputs[token],1]



class Sklearn_Base(object):
    def __init__(self, Base, SuperFunction,function,parameters):
        self.Base = Base
        self.SuperFunction = SuperFunction
        self.function = function
        self.parameters = parameters

    # def call(self):
    #     interface = exec(self.function)

