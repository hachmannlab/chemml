import pandas as pd
import copy

class cheml_Base(object):
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

#####################################################################Script

class PyScript(cheml_Base):
    def legal_IO(self):
        self.legal_inputs = {'df':None,'api':None, 'value': None}
        self.legal_outputs = {'df':None,'api':None, 'value': None}
        requirements = []
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def fit(self):
        inputs = [item for item in self.legal_inputs if not isinstance(self.legal_inputs[item], type(None))]
        for item in inputs:
            exec("%s = self.legal_inputs['%s'][0]"%(item,item))
        for line in sorted(self.parameters.keys()):
            exec(self.parameters[line])
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token == 'df':
                self.Base.send[(self.iblock, token)] = [df, order.count(token)]
            elif token == 'api':
                self.Base.send[(self.iblock, token)] = [api, order.count(token)]
            elif token == 'value':
                self.Base.send[(self.iblock, token)] = [value, order.count(token)]
            else:
                msg = "@Task #%i(%s): asked to send a non valid output token '%s'" % (
                    self.iblock+1,self.SuperFunction,token)
                raise NameError(msg)
        del self.legal_inputs

#####################################################################DataRepresentation

class RDKitFingerprint(cheml_Base):
    def legal_IO(self):
        self.legal_inputs = {'molfile': None}
        self.legal_outputs = {'df': None}
        requirements = ['cheml', 'pandas', 'rdkit']
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def fit(self):
        from cheml.chem import RDKFingerprint
        if 'molfile' in self.parameters:
            molfile = self.type_check('molfile', cheml_type='molfile', req=True, py_type=False)
        elif not isinstance(self.legal_inputs['molfile'],type(None)):
            molfile = self.legal_inputs['molfile'][0]
        else:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.SuperFunction) + 'No molecule file (molfile) has been passed'
            raise TypeError(msg)
        if 'path' in self.parameters:
            path = self.parameters.pop('path')
        else:
            path = None
        if 'arguments' in self.parameters:
            arguments = self.parameters.pop('arguments')
        else:
            arguments = []

        try:
            model = RDKFingerprint(**self.parameters)
            model.MolfromFile(molfile,path,*arguments)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.SuperFunction) + type(err).__name__ + ': ' + err.message
            raise TypeError(msg)
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token == 'df':
                val = model.Fingerprint()
                self.Base.send[(self.iblock, token)] = [val, order.count(token)]
            else:
                msg = "@Task #%i(%s): asked to send a non valid output token '%s'" % (
                self.iblock + 1, self.SuperFunction, token)
                raise NameError(msg)
        del self.legal_inputs

class Dragon(cheml_Base):
    def legal_IO(self):
        self.legal_inputs = {'molfile': None}
        self.legal_outputs = {'df': None}
        requirements = ['cheml', 'pandas', 'lxml', 'Dragon - not a python library']
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def fit(self):
        from cheml.chem import Dragon
        if 'script' in self.parameters:
            script = self.parameters.pop('script')
        else:
            script = 'new'
        if 'output_directory' in self.parameters:
            output_directory = self.parameters.pop('output_directory')
        else:
            output_directory = ''
        output_directory = self.Base.output_directory + '/' + output_directory
        if ('molFile' not in self.parameters or not self.parameters['molFile']):
            self.parameters['molFile'] = self.type_check('molFile', cheml_type='molFile', req=True, py_type=str)
        else:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.SuperFunction) + 'No molecule file (molfile) has been passed'
            raise TypeError(msg)

        try:
            model = Dragon(**self.parameters)
            model.script_wizard(script, output_directory)
            model.run()
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.SuperFunction) + type(err).__name__ + ': ' + err.message
            raise TypeError(msg)
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token == 'df':
                df_path = model.data_path
                df = pd.read_csv(df_path, sep=None, engine='python')
                self.Base.send[(self.iblock, token)] = [df, order.count(token)]
            else:
                msg = "@Task #%i(%s): asked to send a non valid output token '%s'" % (
                    self.iblock + 1, self.SuperFunction, token)
                raise NameError(msg)
        del self.legal_inputs

class CoulombMatrix(cheml_Base):
    def legal_IO(self):
        self.legal_inputs = {}
        self.legal_outputs = {'df': None}
        self.Base.requirements.append('cheml', 'pandas', 'rdkit')

    def fit(self):
        from cheml.chem import CoulombMatrix
        if 'molfile' in self.parameters:
            molfile = self.parameters.pop('molfile')
        else:
            molfile = ''
        if 'path' in self.parameters:
            path = self.parameters.pop('path')
        else:
            path = None
        if 'reader' in self.parameters:
            reader = self.parameters.pop('reader')
        else:
            reader = 'auto'
        if 'skip_lines' in self.parameters:
            skip_lines = self.parameters.pop('skip_lines')
        else:
            skip_lines = [2,0]
        if 'arguments' in self.parameters:
            arguments = self.parameters.pop('arguments')
        else:
            arguments = []

        try:
            model = CoulombMatrix(**self.parameters)
            model.MolfromFile(molfile, path, reader, skip_lines,*arguments)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.SuperFunction) + type(err).__name__ + ': ' + err.message
            raise TypeError(msg)
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token == 'df':
                val = pd.DataFrame(model.Coulomb_Matrix())
                self.Base.send[(self.iblock, token)] = [val, order.count(token)]
            else:
                msg = "@Task #%i(%s): asked to send a non valid output token '%s'" % (
                    self.iblock + 1, self.SuperFunction, token)
                raise NameError(msg)
        del self.legal_inputs

class DistanceMatrix(cheml_Base):
    def legal_IO(self):
        self.legal_inputs = {'df': None}
        self.legal_outputs = {'df': None}
        self.Base.requirements.append('cheml', 'pandas', 'rdkit')

    def fit(self):
        from cheml.chem import DistanceMatrix
        # check inputs
        if isinstance(self.legal_inputs['df'], type(None)):
            msg = '@Task #%i(%s): input data frame is required'%(self.iblock,self.SuperFunction)
            raise IOError(msg)
        try:
            model = DistanceMatrix(**self.parameters)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.SuperFunction) + type(err).__name__ + ': ' + err.message
            raise TypeError(msg)
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token == 'df':
                val = pd.DataFrame(model.transform(self.legal_inputs['df'][0].values))
                self.Base.send[(self.iblock, token)] = [val, order.count(token)]
            else:
                msg = "@Task #%i(%s): asked to send a non valid output token '%s'" % (
                    self.iblock + 1, self.SuperFunction, token)
                raise NameError(msg)
        del self.legal_inputs

#####################################################################Preprocessor

class MissingValues(cheml_Base):
    def legal_IO(self):
        self.legal_inputs = {'df': None}
        self.legal_outputs = {'df': None, 'api':None}
        requirements = ['cheml', 'pandas']
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def fit(self):
        from cheml.preprocessing import missing_values
        df = self.type_check('df', cheml_type='df', req=True, py_type=pd.DataFrame)
        try:
            model = missing_values(**self.parameters)
            df = model.fit(df)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.SuperFunction) + type(err).__name__ + ': ' + err.message
            raise TypeError(msg)
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token == 'df':
                self.Base.send[(self.iblock, token)] = [df, order.count(token)]
            elif token == 'api':
                self.Base.send[(self.iblock, token)] = [model, order.count(token)]
            else:
                msg = "@Task #%i(%s): asked to send a non valid output token '%s'" % (
                self.iblock + 1, self.SuperFunction, token)
                raise NameError(msg)
        del self.legal_inputs

class Trimmer(cheml_Base):
    def legal_IO(self):
        self.legal_inputs = {'dfx': None, 'dfy': None}
        self.legal_outputs = {'df': None, 'api': None}
        requirements = ['cheml', 'pandas']
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def fit(self):
        from cheml.initializtion import Trimmer
        dfx = self.type_check('dfx', cheml_type='dfx', req=True, py_type=pd.DataFrame)
        dfy = self.type_check('dfy', cheml_type='dfy', req=True, py_type=pd.DataFrame)
        try:
            model = Trimmer(**self.parameters)
            dfx, dfy = model.fit_Transform(dfx,dfy)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.SuperFunction) + type(
                err).__name__ + ': ' + err.message
            raise TypeError(msg)
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token == 'dfx':
                self.Base.send[(self.iblock, token)] = [dfx, order.count(token)]
            elif token == 'dfy':
                self.Base.send[(self.iblock, token)] = [dfy, order.count(token)]
            elif token == 'api':
                self.Base.send[(self.iblock, token)] = [model, order.count(token)]
            else:
                msg = "@Task #%i(%s): asked to send a non valid output token '%s'" % (
                    self.iblock + 1, self.SuperFunction, token)
                raise NameError(msg)
        del self.legal_inputs

class Uniformer(cheml_Base):
    def legal_IO(self):
        self.legal_inputs = {'dfx': None, 'dfy': None}
        self.legal_outputs = {'df': None, 'api': None}
        requirements = ['cheml', 'pandas']
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def fit(self):
        from cheml.initializtion import Uniformer
        dfx = self.type_check('dfx', cheml_type='dfx', req=True, py_type=pd.DataFrame)
        dfy = self.type_check('dfy', cheml_type='dfy', req=True, py_type=pd.DataFrame)
        try:
            model = Trimmer(**self.parameters)
            dfx, dfy = model.fit_Transform(dfx, dfy)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.SuperFunction) + type(
                err).__name__ + ': ' + err.message
            raise TypeError(msg)
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token == 'dfx':
                self.Base.send[(self.iblock, token)] = [dfx, order.count(token)]
            elif token == 'dfy':
                self.Base.send[(self.iblock, token)] = [dfy, order.count(token)]
            elif token == 'api':
                self.Base.send[(self.iblock, token)] = [model, order.count(token)]
            else:
                msg = "@Task #%i(%s): asked to send a non valid output token '%s'" % (
                    self.iblock + 1, self.SuperFunction, token)
                raise NameError(msg)
        del self.legal_inputs

#####################################################################Input

class File(cheml_Base):
    def legal_IO(self):
        self.legal_inputs = {}
        self.legal_outputs = {'df':None}
        requirements = ['cheml','pandas']
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def fit(self):
        from cheml.initialization import File
        try:
            df = File(**self.parameters)
        except Exception as err:
            msg = '@Task #%i(%s): '%(self.iblock+1, self.SuperFunction) + type(err).__name__ + ': '+ err.message
            raise TypeError(msg)
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token == 'df':
                val = df
                self.Base.send[(self.iblock, token)] = [val, order.count(token)]
            else:
                msg = "@Task #%i(%s): asked to send a non valid output token '%s'" % (
                    self.iblock+1,self.SuperFunction,token)
                raise NameError(msg)
        del self.legal_inputs

class Merge(cheml_Base):
    def legal_IO(self):
        self.legal_inputs = {'df1':None, 'df2':None}
        self.legal_outputs = {'df':None}
        requirements = ['cheml','pandas']
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def fit(self):
        from cheml.initialization import Merge
        # check inputs
        if isinstance(self.legal_inputs['df1'], type(None)) or isinstance(self.legal_inputs['df2'], type(None)):
            msg = '@Task #%i(%s): both inputs (df1 and df2) are required'%(self.iblock,self.SuperFunction)
            raise IOError(msg)
        try:
            df = Merge(self.legal_inputs['df1'][0], self.legal_inputs['df2'][0])
        except Exception as err:
            msg = '@Task #%i(%s): '%(self.iblock+1, self.SuperFunction) + type(err).__name__ + ': '+ err.message
            raise TypeError(msg)
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token == 'df':
                val = df
                self.Base.send[(self.iblock, token)] = [val, order.count(token)]
            else:
                msg = "@Task #%i(%s): asked to send a non valid output token '%s'" % (self.iblock+1,self.SuperFunction,token)
                raise NameError(msg)
        del self.legal_inputs

class Split(cheml_Base):
    def legal_IO(self):
        self.legal_inputs = {'df':None}
        self.legal_outputs = {'df1':None, 'df2':None}
        requirements = ['cheml','pandas']
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def fit(self):
        from cheml.initialization import Split
        # check inputs
        if not isinstance(self.legal_inputs['df'], type(None)):
            self.parameters['X'] = self.type_check('df', cheml_type='df', req=True, py_type=pd.DataFrame)
        else:
            msg = '@Task #%i(%s): an input data frame is required'%(self.iblock,self.SuperFunction)
            raise IOError(msg)
        try:
            df1, df2 = Split(**self.parameters)
        except Exception as err:
            msg = '@Task #%i(%s): '%(self.iblock+1, self.SuperFunction) + type(err).__name__ + ': '+ err.message
            raise TypeError(msg)
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token == 'df1':
                val = df1
                self.Base.send[(self.iblock, token)] = [val, order.count(token)]
            elif token == 'df2':
                val = df2
                self.Base.send[(self.iblock, token)] = [val, order.count(token)]
            else:
                msg = "@Task #%i(%s): asked to send a non valid output token '%s'" % (self.iblock+1,self.SuperFunction,token)
                raise NameError(msg)
        del self.legal_inputs
#####################################################################Output

class SaveFile(cheml_Base):
    def legal_IO(self):
        self.legal_inputs = {'df': None}
        self.legal_outputs = {'filepath': None}
        requirements = ['cheml','pandas']
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def fit(self):
        from cheml.initialization import SaveFile
        # check inputs
        if isinstance(self.legal_inputs['df'], type(None)):
            msg = '@Task #%i(%s): input data frame is required'%(self.iblock,self.SuperFunction)
            raise IOError(msg)
        try:
            model = SaveFile(**self.parameters)
            model.fit(self.legal_inputs['df'][0],self.Base.output_directory)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.SuperFunction) + type(err).__name__ + ': ' + err.message
            raise TypeError(msg)
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token == 'filepath':
                val = model.file_path
                self.Base.send[(self.iblock, token)] = [val, order.count(token)]
            else:
                msg = "@Task #%i(%s): asked to send a non valid output token '%s'" % (self.iblock+1,self.SuperFunction,token)
                raise NameError(msg)
        del self.legal_inputs

#####################################################################Regression

class nn_psgd(cheml_Base):
    def legal_IO(self):
        self.legal_inputs = {'dfx_train': None, 'dfx_test': None, 'dfy_train': None, 'dfy_test': None}
        self.legal_outputs = {'dfy_train_pred': None, 'model': None}
        requirements = ['cheml', 'pandas']
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def fit(self):
        from cheml.nn import nn_psgd
        cheml_type = "%s_%s" % (self.Base.graph_info[self.iblock][0], self.Base.graph_info[self.iblock][1])
        self.Base.cheml_type['regressor'].append(cheml_type)
        dfx_train = self.type_check('dfx_train', cheml_type='df', req=True, py_type=pd.DataFrame).values
        dfx_test = self.type_check('dfx_test', cheml_type='df', req=True, py_type=pd.DataFrame).values
        dfy_train = self.type_check('dfy_train', cheml_type='df', req=True, py_type=pd.DataFrame)
        dfy_header = dfy_train.columns
        dfy_train = dfy_train.values
        dfy_test = self.type_check('dfy_test', cheml_type='df', req=True, py_type=pd.DataFrame).values

        try:
            model = nn_psgd.train(dfx_train,dfx_test,dfy_train,dfy_test,**self.parameters)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.SuperFunction) + type(
                err).__name__ + ': ' + err.message
            raise TypeError(msg)

        dfy_pred = nn_psgd.output(dfx_train,model)
        dfy_pred = pd.DataFrame(dfy_pred, columns=dfy_header)

        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token == 'model':
                self.Base.send[(self.iblock, token)] = [model, order.count(token)]
            elif token == 'dfy_train_pred':
                self.Base.send[(self.iblock, token)] = [dfy_pred, order.count(token)]
            else:
                msg = "@Task #%i(%s): non valid output token '%s'" % (self.iblock + 1, self.SuperFunction, token)
                raise NameError(msg)
        del self.legal_inputs

class nn_dsgd(cheml_Base):
    # must be run with slurm script
    # Todo: first fix the slurm script function at cheml.initialization
    # Todo: then embede the slurm commands in this class to run the slurm script
    # Todo: or make the slurm script in this function too
    def legal_IO(self):
        self.legal_inputs = {'dfx_train': None, 'dfx_test': None, 'dfy_train': None, 'dfy_test': None}
        self.legal_outputs = {'dfy_train_pred': None, 'model': None}
        requirements = ['cheml', 'pandas']
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def fit(self):
        from cheml.nn import nn_dsgd
        cheml_type = "%s_%s" % (self.Base.graph_info[self.iblock][0], self.Base.graph_info[self.iblock][1])
        self.Base.cheml_type['regressor'].append(cheml_type)
        dfx_train = self.type_check('dfx_train', cheml_type='df', req=True, py_type=pd.DataFrame).values
        dfx_test = self.type_check('dfx_test', cheml_type='df', req=True, py_type=pd.DataFrame).values
        dfy_train = self.type_check('dfy_train', cheml_type='df', req=True, py_type=pd.DataFrame)
        dfy_header = dfy_train.columns
        dfy_train = dfy_train.values
        dfy_test = self.type_check('dfy_test', cheml_type='df', req=True, py_type=pd.DataFrame).values

        try:
            model = nn_psgd.train(dfx_train,dfx_test,dfy_train,dfy_test,**self.parameters)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.SuperFunction) + type(
                err).__name__ + ': ' + err.message
            raise TypeError(msg)

        dfy_pred = nn_psgd.output(dfx_train,model)
        dfy_pred = pd.DataFrame(dfy_pred, columns=dfy_header)

        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token == 'model':
                self.Base.send[(self.iblock, token)] = [model, order.count(token)]
            elif token == 'dfy_train_pred':
                self.Base.send[(self.iblock, token)] = [dfy_pred, order.count(token)]
            else:
                msg = "@Task #%i(%s): non valid output token '%s'" % (self.iblock + 1, self.SuperFunction, token)
                raise NameError(msg)
        del self.legal_inputs