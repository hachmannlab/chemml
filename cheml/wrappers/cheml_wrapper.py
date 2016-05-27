from .sct_utils import isfloat, islist, istuple, isnpdot, std_datetime_str

class cheml_Base(object):
    """
    Do not instantiate this class
    """
    def __init__(self, Base,parameters,iblock):
        self.Base = Base
        self.parameters = parameters
        self.iblock = iblock

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
                    msg = '@function #%i: only one input per each available input can be received.' % self.iblock + 1
                    raise IOError(msg)
            else:
                msg = "@function #%i: received a non valid input token '%s', sent by function #%i" % (
                self.iblock + 1, edge[3], edge[0] + 1)
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
                msg = '@function #%i: broken pipe in edge %s - nothing has been sent' % (self.iblock + 1, str(edge))
                raise IOError(msg)
        return self.legal_inputs

    def send(self):
        send = [edge for edge in self.Base.graph if edge[0] == self.iblock]
        for edge in send:
            key = edge[0:1]
            if key in self.Base.send:
                self.Base.send[key][1] += 1
            else:
                self.Base.send[key] = [self.legal_outputs[edge[1]], 1]


class File(cheml_Base):
    def legal_IO(self):
        self.legal_inputs = {}
        self.legal_outputs = {'df':None}
        self.Base.requirements.append('scikit_learn')

    def fit(self):
        from cheml.initialization import File
        try:
            df = File(**parametrs)
        except Exception as err:
            msg = 'built in error in function #%i: '%iblock + type(err).__name__ + ': '+ err.message
            raise TypeError(msg)
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in order:
            if token == 'df':
                self.legal_outputs[token] = df
            else:
                msg = "asked to send a non valid output token '%s' in function #%i" % (token, self.iblock + 1)
                raise NameError(msg)

