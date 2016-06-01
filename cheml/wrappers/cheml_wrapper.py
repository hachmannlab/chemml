import pandas as pd

class cheml_Base(object):
    """
    Do not instantiate this class
    """
    def __init__(self, Base,parameters,iblock,SuperFunction):
        self.Base = Base
        self.parameters = parameters
        self.iblock = iblock
        self.SuperFunction = SuperFunction

    def run(self):
        self.legal_IO()
        self.receive()
        self.fit()
        self.send()

    def receive(self):
        recv = [edge for edge in self.Base.graph if edge[2] == self.iblock]
        self.Base.graph = [edge for edge in self.Base.graph if edge[2] != self.iblock]
        # check received tokens
        count = [0] * len(self.legal_inputs)
        for edge in recv:
            if edge[3] in self.legal_inputs:
                ind = self.legal_inputs.index(edge[2])
                count[ind] += 1
                if count[ind] > 1:
                    msg = '@Task #%i(%s): only one input per each available input can be received.' % (
                        self.iblock+1,self.SuperFunction)
                    raise IOError(msg)
            else:
                msg = "@Task #%i(%s): received a non valid input token '%s', sent by function #%i" % (
                    self.iblock+1,self.SuperFunction,edge[3], edge[0] + 1)
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
                msg = '@Task #%i(%s): broken pipe in edge %s - nothing has been sent' % (
                    self.iblock+1,self.SuperFunction,str(edge))
                raise IOError(msg)
        return self.legal_inputs

    def type_check(self,token,specific_type=None):
        if token[0:2]=='df':
            if isinstance(self.legal_inputs[token],pd.DataFrame):
                return self.legal_inputs[token]
            else:
                return None
        elif token[0:3]=='api':
            if isinstance(self.legal_inputs[token], specific_type):
                return self.legal_inputs[token]
            else:
                return None

    def send(self):
        send = [edge for edge in self.Base.graph if edge[0] == self.iblock]
        for edge in send:
            # insert type_check for send and receive control
            key = edge[0:1]
            if key in self.Base.send:
                self.Base.send[key][1] += 1
            else:
                self.Base.send[key] = [self.legal_outputs[edge[1]], 1]


class File(cheml_Base):
    def legal_IO(self):
        self.legal_inputs = {}
        self.legal_outputs = {'df':None}
        self.Base.requirements.append('cheml','pandas')

    def fit(self):
        from cheml.initialization import File
        try:
            df = File(**self.parameters)
        except Exception as err:
            msg = '@Task #%i(%s): '%(self.iblock+1, self.SuperFunction) + type(err).__name__ + ': '+ err.message
            raise TypeError(msg)
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in order:
            if token == 'df':
                self.legal_outputs[token] = df
            else:
                msg = "@Task #%i(%s): asked to send a non valid output token '%s'" % (
                    self.iblock+1,self.SuperFunction,token)
                raise NameError(msg)

class Merge(cheml_Base):
    def legal_IO(self):
        self.legal_inputs = {'df1':None, 'df2':None}
        self.legal_outputs = {'df':None}
        self.Base.requirements.append('cheml','pandas')

    def fit(self):
        from cheml.initialization import Merge
        # check inputs
        if self.legal_inputs['df1'] == None or self.legal_inputs['df2'] == None:
            msg = '@Task #%i(%s): both inputs (df1 and df2) are required'%(self.iblock,self.SuperFunction)
            raise IOError(msg)
        try:
            df = Merge(self.legal_inputs['df1'], self.legal_inputs['df2'])
        except Exception as err:
            msg = '@Task #%i(%s): '%(self.iblock+1, self.SuperFunction) + type(err).__name__ + ': '+ err.message
            raise TypeError(msg)
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in order:
            if token == 'df':
                self.legal_outputs[token] = df
            else:
                msg = "@Task #%i(%s): asked to send a non valid output token '%s'" % (self.iblock+1,self.SuperFunction,token)
                raise NameError(msg)

class Split(cheml_Base):
    def legal_IO(self):
        self.legal_inputs = {'df':None}
        self.legal_outputs = {'df1':None, 'df2':None}
        self.Base.requirements.append('cheml','pandas')

    def fit(self):
        from cheml.initialization import Merge
        # check inputs
        if self.legal_inputs['df'] == None:
            msg = '@Task #%i(%s): an input data frame is required'%(self.iblock,self.SuperFunction)
            raise IOError(msg)
        else:
            self.parameters['X'] = self.legal_inputs['df']
        try:
            df1, df2 = Merge(**self.parameters)
        except Exception as err:
            msg = '@Task #%i(%s): '%(self.iblock+1, self.SuperFunction) + type(err).__name__ + ': '+ err.message
            raise TypeError(msg)
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in order:
            if token == 'df1':
                self.legal_outputs[token] = df1
            elif token == 'df2':
                self.legal_outputs[token] = df2
            else:
                msg = "@Task #%i(%s): asked to send a non valid output token '%s'" % (self.iblock+1,self.SuperFunction,token)
                raise NameError(msg)
