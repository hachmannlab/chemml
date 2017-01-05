import pandas as pd
import numpy as np

from .base import BASE

#####################################################################Script

class PyScript(BASE):
    def legal_IO(self):
        self.legal_inputs = {'var1':None,'var2':None,'var3':None, 'var4':None, 'var5': None}
        self.legal_outputs = {'var_out1':None,'var_out2':None,'var_out3':None,'var_out4':None,'var_out5':None}
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
            if token == 'var_out1':
                self.Base.send[(self.iblock, token)] = [var_out1, order.count(token)]
            elif token == 'var_out2':
                self.Base.send[(self.iblock, token)] = [var_out2, order.count(token)]
            elif token == 'var_out3':
                self.Base.send[(self.iblock, token)] = [var_out3, order.count(token)]
            elif token == 'var_out4':
                self.Base.send[(self.iblock, token)] = [var_out4, order.count(token)]
            elif token == 'var_out5':
                self.Base.send[(self.iblock, token)] = [var_out5, order.count(token)]
            else:
                msg = "@Task #%i(%s): asked to send a non valid output token '%s'" % (
                    self.iblock+1,self.SuperFunction,token)
                raise NameError(msg)
        del self.legal_inputs

#####################################################################DataRepresentation

class RDKitFingerprint(BASE):
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

class Dragon(BASE):
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
        if ('molFile' not in self.parameters or self.parameters['molFile'] is None):
            self.parameters['molFile'] = self.type_check('molfile', cheml_type='descriptor', req=True, py_type=str)
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
                df = df.drop(['No.','NAME'],axis=1)
                self.Base.send[(self.iblock, token)] = [df, order.count(token)]
            else:
                msg = "@Task #%i(%s): asked to send a non valid output token '%s'" % (
                    self.iblock + 1, self.SuperFunction, token)
                raise NameError(msg)
        del self.legal_inputs

class CoulombMatrix(BASE):
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

class DistanceMatrix(BASE):
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

class MissingValues(BASE):
    def legal_IO(self):
        self.legal_inputs = {'dfx': None, 'dfy':None}
        self.legal_outputs = {'dfx': None, 'dfy': None, 'api':None}
        requirements = ['cheml',  'pandas']
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def fit(self):
        from cheml.preprocessing import missing_values
        dfx = self.type_check('dfx', cheml_type='df', req=True, py_type=pd.DataFrame)
        try:
            model = missing_values(**self.parameters)
            dfx = model.fit_transform(dfx)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.SuperFunction) + type(err).__name__ + ': ' + err.message
            raise TypeError(msg)
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token == 'dfx':
                self.Base.send[(self.iblock, token)] = [dfx, order.count(token)]
            elif token == 'dfy':
                dfy = self.type_check('dfy', cheml_type='df', req=True, py_type=pd.DataFrame)
                dfy = model.transform(dfy)
                self.Base.send[(self.iblock, token)] = [dfy, order.count(token)]
            elif token == 'api':
                self.Base.send[(self.iblock, token)] = [model, order.count(token)]
            else:
                msg = "@Task #%i(%s): asked to send a non valid output token '%s'" % (
                self.iblock + 1, self.SuperFunction, token)
                raise NameError(msg)
        del self.legal_inputs

class Trimmer(BASE):
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
            dfx, dfy = model.fit_transform(dfx,dfy)
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

class Uniformer(BASE):
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
            model = Uniformer(**self.parameters)
            dfx, dfy = model.fit_transform(dfx, dfy)
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

class Constant(BASE):
    def legal_IO(self):
        self.legal_inputs = {'df': None}
        self.legal_outputs = {'df': None, 'api': None, 'removed_columns_': None}
        requirements = ['cheml', 'pandas']
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def fit(self):
        from cheml.preprocessing import Constant
        df = self.type_check('df', cheml_type='df', req=True, py_type=pd.DataFrame)
        try:
            model = Constant()
            df = model.fit_transform(df)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.SuperFunction) + type(
                err).__name__ + ': ' + err.message
            raise TypeError(msg)
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token == 'df':
                self.Base.send[(self.iblock, token)] = [df, order.count(token)]
            elif token == 'removed_columns_':
                val = pd.DataFrame(model.removed_columns_)
                self.Base.send[(self.iblock, token)] = [val, order.count(token)]
            elif token == 'api':
                self.Base.send[(self.iblock, token)] = [model, order.count(token)]
            else:
                msg = "@Task #%i(%s): asked to send a non valid output token '%s'" % (
                    self.iblock + 1, self.SuperFunction, token)
                raise NameError(msg)
        del self.legal_inputs

#####################################################################Input

class File(BASE):
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

class Merge(BASE):
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

class Split(BASE):
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

class SaveFile(BASE):
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

class NN_PSGD(BASE):
    def legal_IO(self):
        self.legal_inputs = {'dfx_train': None, 'dfx_test': None, 'dfy_train': None, 'dfy_test': None}
        self.legal_outputs = {'dfy_train_pred': None, 'model': None}
        requirements = ['cheml', 'pandas']
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def fit(self):
        from cheml.nn import nn_psgd
        cheml_type = "%s_%s" % (self.Base.graph_info[self.iblock][0], self.Base.graph_info[self.iblock][1])
        self.Base.cheml_type['regressor'].append(cheml_type)
        if self.legal_inputs['dfx_train'] is not None:
            dfx_train = self.type_check('dfx_train', cheml_type='df', req=True, py_type=pd.DataFrame).values
            dfx_test = self.type_check('dfx_test', cheml_type='df', req=True, py_type=pd.DataFrame).values
            dfy_train = self.type_check('dfy_train', cheml_type='df', req=True, py_type=pd.DataFrame)
            dfy_header = dfy_train.columns
            dfy_train = dfy_train.values
            print dfy_train
            dfy_test = self.type_check('dfy_test', cheml_type='df', req=True, py_type=pd.DataFrame).values
            try:
                model = nn_psgd.train(dfx_train,dfx_test,dfy_train,dfy_test,**self.parameters)
            except Exception as err:
                msg = '@Task #%i(%s): ' % (self.iblock + 1, self.SuperFunction) + type(
                    err).__name__ + ': ' + err.message
                raise TypeError(msg)
            dfy_pred = nn_psgd.output(dfx_train,model)
            dfy_pred = pd.DataFrame(dfy_pred, columns=dfy_header)
        else:
            model = self.parameters

        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token == 'model':
                self.Base.send[(self.iblock, token)] = [model, order.count(token)]
            elif token == 'dfy_train_pred':
                if self.legal_inputs['dfx_train'] is None:
                    msg = "@Task #%i(%s): No input data" % (self.iblock + 1, self.SuperFunction)
                    raise NameError(msg)
                self.Base.send[(self.iblock, token)] = [dfy_pred, order.count(token)]
            else:
                msg = "@Task #%i(%s): non valid output token '%s'" % (self.iblock + 1, self.SuperFunction, token)
                raise NameError(msg)
        del self.legal_inputs

class nn_dsgd(BASE):
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