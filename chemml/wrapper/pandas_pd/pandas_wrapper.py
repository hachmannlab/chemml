import os
import pandas as pd


from chemml.wrapper.base import BASE

##################################################################### 1 Enter Data

# input

class read_table(BASE):
    def fit(self):
        # step1: check inputs
        # step2: assign inputs to parameters if necessary (param = @token)
        self.paramFROMinput()
        # step3: check the dimension of input data frame
        # step4: import module and make APIs
        try:
            df = pd.read_table(**self.parameters)
        except Exception as err:
            msg = '@Task #%i(%s): '%(self.iblock+1, self.Task) + type(err).__name__ + ': '+ err.message
            raise TypeError(msg)

        # step5: process
        print ('(rows, columns): ', df.shape)
        # if 'header' in self.parameters and self.parameters['header'] is not None:
        #     print 'headers: ', list(df.columns)

        # step6: send out
        order = [edge[1] for edge in self.Base.graph if edge[0]==self.iblock]
        for token in set(order):
            if token not in self.outputs:
                msg = "@Task #%i(%s): not a valid output token '%s'" % (self.iblock + 1, self.Task, token)
                raise NameError(msg)
            elif token == 'df':
                self.set_value(token, df)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]


        # step7: delete all inputs from memory
        del self.inputs

class read_excel(BASE):
    def fit(self):
        # print(self)
        # step1: check inputs
        # step2: assign inputs to parameters if necessary (param = @token)
        self.paramFROMinput()
        # for param in self.parameters:
            # print(param)
            # print(self.parameters[param])
        # print(self.paramFROMinput())
        # step3: check the dimension of input data frame
        # step4: import module and make APIs
        try:
            df = pd.read_excel(**self.parameters)
        except Exception as err:
            msg = '@Task #%i(%s): '%(self.iblock+1, self.Task) + type(err).__name__ + ': '+ err.message
            raise TypeError(msg)

        # step5: process
        # print ('(rows, columns): ', df.shape)
        # if 'header' in self.parameters and self.parameters['header'] is not None:
        #     print 'header: ', df.columns

        # step6: send out
        order = [edge[1] for edge in self.Base.graph if edge[0]==self.iblock]
        for token in set(order):
            if token not in self.outputs:
                msg = "@Task #%i(%s): not a valid output token '%s'" % (self.iblock + 1, self.Task, token)
                raise NameError(msg)
            elif token == 'df':
                self.set_value(token, df)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]

        # step7: delete all inputs from memory
        del self.inputs


# Prepare
class concat(BASE):
    def fit(self):
        # step1: check inputs
        objs = []
        dfs = ['df1','df2','df3']
        for f in dfs:
            df = self.inputs[f].value
            if isinstance(df,pd.DataFrame):
                objs.append(df)
        if len(objs) == 0:
            msg = "@Task #%i(%s): All the input dataframes are None objects" % (self.iblock + 1, self.Task)
            raise IOError(msg)

        # step2: assign inputs to parameters if necessary (param = @token)
        # self.paramFROMinput()
        # step3: check the dimension of input data frame
        # step4: import module and make APIs
        try:
            df_out = pd.concat(objs,**self.parameters)
        except Exception as err:
            msg = '@Task #%i(%s): '%(self.iblock+1, self.Task) + type(err).__name__ + ': '+ err.message
            raise TypeError(msg)

        # step5: process
        # step6: send out
        order = [edge[1] for edge in self.Base.graph if edge[0]==self.iblock]
        for token in set(order):
            if token not in self.outputs:
                msg = "@Task #%i(%s): not a valid output token '%s'" % (self.iblock + 1, self.Task, token)
                raise NameError(msg)
            elif token == 'df':
                self.set_value(token, df_out)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]

        # step7: delete all inputs from memory
        del self.inputs


# Search
class corr(BASE):
    def fit(self):
        # step1: check inputs
        self.required('df', True)
        df = self.inputs['df'].value
        # step2: assign inputs to parameters if necessary (param = @token)
        self.paramFROMinput()
        # step3: check the dimension of input data frame
        # step4: import module and make APIs
        try:
            df_out = df.corr(**self.parameters)
        except Exception as err:
            msg = '@Task #%i(%s): '%(self.iblock+1, self.Task) + type(err).__name__ + ': '+ err.message
            raise TypeError(msg)

        # step5: process
        # step6: send out
        order = [edge[1] for edge in self.Base.graph if edge[0]==self.iblock]
        for token in set(order):
            if token not in self.outputs:
                msg = "@Task #%i(%s): not a valid output token '%s'" % (self.iblock + 1, self.Task, token)
                raise NameError(msg)
            elif token == 'df':
                self.set_value(token, df_out)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]

        # step7: delete all inputs from memory
        del self.inputs

# Visualize
class plot(BASE):
    def fit(self):
        # step1: check inputs
        self.required('df', True)
        df = self.inputs['df'].value
        # step2: assign inputs to parameters if necessary (param = @token)
        # self.paramFROMinput()
        # step3: check the dimension of input data frame
        # step4: import module and make APIs
        try:
            ax = df.plot(**self.parameters)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(err).__name__ + ': ' + err.message
            raise TypeError(msg)

        # step5: process
        # step6: send out
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token not in self.outputs:
                msg = "@Task #%i(%s): not a valid output token '%s'" % (self.iblock + 1, self.Task, token)
                raise NameError(msg)
            elif token == 'fig':
                self.set_value(token, ax)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]

        # step7: delete all inputs from memory
        del self.inputs


# Store
class to_csv(BASE):
    def fit(self):
        # step1: check inputs
        self.required('df', True)
        df = self.inputs['df'].value
        # step2: assign inputs to parameters if necessary (param = @token)
        # self.paramFROMinput()
        if 'path_or_buf' in self.parameters:
            if isinstance(self.parameters['path_or_buf'], str):
                self.parameters['path_or_buf'] = os.path.join(self.Base.output_directory,self.parameters['path_or_buf'])
        # step3: check the dimension of input data frame
        # step4: import module and make APIs
        try:
            df.to_csv(**self.parameters)
        except Exception as err:
            msg = '@Task #%i(%s): '%(self.iblock+1, self.Task) + type(err).__name__ + ': '+ err.message
            raise TypeError(msg)

        # step5: process
        print ('(rows, columns): ', df.shape)
        # if 'header' in self.parameters and self.parameters['header'] is not None:
        #     print 'headers: ', list(df.columns)

        # step6: send out
        order = [edge[1] for edge in self.Base.graph if edge[0]==self.iblock]
        for token in set(order):
            if token not in self.outputs:
                msg = "@Task #%i(%s): not a valid output token '%s'" % (self.iblock + 1, self.Task, token)
                raise NameError(msg)
            elif token == 'df':
                self.set_value(token, df)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]


        # step7: delete all inputs from memory
        del self.inputs