import pandas as pd

from ..base import BASE, LIBRARY

##################################################################### 1 Enter Data

# input

class read_table(BASE, LIBRARY):
    def legal_IO(self):
        self.legal_inputs = {}
        self.legal_outputs = {'df': None}
        requirements = ['pandas']
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def fit(self):
        # step1: check inputs
        # step2: assign inputs to parameters if necessary (param = @token)
        # step3: check the dimension of input data frame
        # step4: import module and make APIs
        try:
            df = pd.read_table(**self.parameters)
        except Exception as err:
            msg = '@Task #%i(%s): '%(self.iblock+1, self.SuperFunction) + type(err).__name__ + ': '+ err.message
            raise TypeError(msg)

        # step5: process
        print '(rows, columns): ', df.shape
        if 'header' in self.parameters and self.parameters['header'] is not None:
            print 'headers: ', list(df.columns)

        # step6: send out
        order = [edge[1] for edge in self.Base.graph if edge[0]==self.iblock]
        for token in set(order):
            if token == 'df':
                self.Base.send[(self.iblock, token)] = [df, order.count(token),
                                                        (self.iblock, token, self.Host, self.Function)]
            else:
                msg = "@Task #%i(%s): non valid output token '%s'" % (self.iblock+1, self.SuperFunction, token)
                raise NameError(msg)

        # step7: delete all inputs from memory
        del self.legal_inputs

class read_excel(BASE, LIBRARY):
    def legal_IO(self):
        self.legal_inputs = {}
        self.legal_outputs = {'df': None}
        requirements = ['pandas']
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def fit(self):
        # step1: check inputs
        # step2: assign inputs to parameters if necessary (param = @token)
        # step3: check the dimension of input data frame
        # step4: import module and make APIs
        try:
            df = pd.read_excel(**self.parameters)
        except Exception as err:
            msg = '@Task #%i(%s): '%(self.iblock+1, self.SuperFunction) + type(err).__name__ + ': '+ err.message
            raise TypeError(msg)

        # step5: process
        print '(rows, columns): ', df.shape
        if 'header' in self.parameters and self.parameters['header'] is not None:
            print 'header: ', df.columns

        # step6: send out
        order = [edge[1] for edge in self.Base.graph if edge[0]==self.iblock]
        for token in set(order):
            if token == 'df':
                self.Base.send[(self.iblock, token)] = [df, order.count(token),
                                                        (self.iblock, token, self.Host, self.Function)]
            else:
                msg = "@Task #%i(%s): non valid output token '%s'" % (self.iblock+1, self.SuperFunction, token)
                raise NameError(msg)

        # step7: delete all inputs from memory
        del self.legal_inputs


