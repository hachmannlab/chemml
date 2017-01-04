class BASE(object):
    """
    Do not instantiate this class
    """
    def __init__(self, Base, parameters, iblock, SuperFunction):
        self.Base = Base
        self.parameters = parameters
        self.iblock = iblock
        self.SuperFunction = SuperFunction

    def run(self):
        self.legal_IO()
        self.receive()
        self.fit()

    def receive(self):
        recv = [edge for edge in self.Base.graph if edge[2] == self.iblock]
        self.Base.graph = tuple([edge for edge in self.Base.graph if edge[2] != self.iblock])
        # check received tokens to (1) be a legal input, and (2) be unique.
        count = {token: 0 for token in self.legal_inputs}
        for edge in recv:
            if edge[3] in self.legal_inputs:
                count[edge[3]] += 1
                if count[edge[3]] > 1:
                    msg = '@Task #%i(%s): only one input per each available input path/token can be received.' % (
                        self.iblock + 1, self.SuperFunction)
                    raise IOError(msg)
            else:
                msg = "@Task #%i(%s): received a non valid input token '%s', sent by function #%i" % (
                    self.iblock + 1, self.SuperFunction, edge[3], edge[0] + 1)
                raise IOError(msg)
        for edge in recv:
            key = edge[0:2]
            if key in self.Base.send:
                if self.Base.send[key][1] > 0:
                    value = self.Base.send[key][0]
                    # TODO: deepcopy is memory consuming
                #     value = copy.deepcopy(self.Base.send[key][0])
                # else:
                #     value = self.Base.send[key][0]
                # Todo: informative token should be a list of (int(edge[0],edge[1])
                informative_token = (int(edge[0]), edge[1]) + self.Base.graph_info[int(edge[0])]
                self.legal_inputs[edge[3]] = (value, informative_token)
                del value
                self.Base.send[key][1] -= 1
                if self.Base.send[key][1] == 0:
                    del self.Base.send[key]
            else:
                msg = '@Task #%i(%s): broken pipe in token %s - nothing has been sent' % (
                    self.iblock + 1, self.SuperFunction, edge[3])
                raise IOError(msg)
        return self.legal_inputs

    def _error_type(self, token):
        msg = "@Task #%i(%s): The type of input with token '%s' is not valid" \
              % (self.iblock + 1, self.SuperFunction, token)
        raise IOError(msg)

    def type_check(self, token, cheml_type, req=False, py_type=False):
        if isinstance(self.legal_inputs[token], type(None)):
            if req:
                msg = "@Task #%i(%s): The input type with token '%s' is required." \
                      % (self.iblock + 1, self.SuperFunction, token)
                raise IOError(msg)
            else:
                return None
        else:
            slit0 = self.legal_inputs[token][0]
            slit1 = self.legal_inputs[token][1]
            if py_type:
                if not isinstance(slit0, py_type):
                    self._error_type(token)
            # if cheml_type == 'df':
            #     if not slit1[1][0:2] == 'df':
            #         self._error_type(token)
            # elif cheml_type == 'regressor':
            #     if slit1[2] + '_' + slit1[3] not in self.Base.cheml_type['regressor']:
            #         self._error_type(token)
            # elif cheml_type == 'preprocessor':
            #     if slit1[2] + '_' + slit1[3] not in self.Base.cheml_type['preprocessor']:
            #         self._error_type(token)
            # elif cheml_type == 'divider':
            #     if slit1[2] + '_' + slit1[3] not in self.Base.cheml_type['divider']:
            #         self._error_type(token)
            # else:
            #     msg = "@Task #%i(%s): The type of input with token '%s' must be %s not %s" \
            #           % (self.iblock + 1, self.SuperFunction, token, str(py_type), str(type(slit0)))
            #     raise IOError(msg)
            return slit0
