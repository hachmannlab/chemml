import pandas as pd
import numpy as np
import cheml
import os

from ..base import BASE, LIBRARY

#####################################################################Script

class PyScript(BASE, LIBRARY):
    def legal_IO(self):
        self.legal_inputs = {'df1':None,'df2':None,'api1':None, 'api2':None, 'var1': None, 'var2': None}
        self.legal_outputs = {'df_out1':None,'df_out2':None,'api_out1':None,'api_out2':None,'var_out1':None,'var_out2':None}
        requirements = []
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def fit(self):
        # step1: check inputs
        df1, df1_info = self.input_check('df1', req=False, py_type=pd.DataFrame)
        df2, df2_info = self.input_check('df2', req=False, py_type=pd.DataFrame)
        inputs = [item for item in self.legal_inputs if not isinstance(self.legal_inputs[item], type(None))]
        for item in inputs:
            exec("%s = self.legal_inputs['%s'][0]"%(item,item))
        for line in sorted(self.parameters.keys()):
            exec(self.parameters[line])
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token == 'df_out1':
                self.Base.send[(self.iblock, token)] = [df_out1, order.count(token)]
            elif token == 'df_out2':
                self.Base.send[(self.iblock, token)] = [df_out2, order.count(token)]
            elif token == 'api_out1':
                self.Base.send[(self.iblock, token)] = [api_out1, order.count(token)]
            elif token == 'api_out2':
                self.Base.send[(self.iblock, token)] = [api_out2, order.count(token)]
            elif token == 'var_out1':
                self.Base.send[(self.iblock, token)] = [var_out1, order.count(token)]
            elif token == 'var_out2':
                self.Base.send[(self.iblock, token)] = [var_out2, order.count(token)]
            else:
                msg = "@Task #%i(%s): asked to send a non valid output token '%s'" % (
                    self.iblock+1,self.SuperFunction,token)
                raise NameError(msg)
        del self.legal_inputs

#####################################################################DataRepresentation

class RDKitFingerprint(BASE,LIBRARY):
    def legal_IO(self):
        self.legal_inputs = {'molfile': None}
        self.legal_outputs = {'df': None, 'removed_rows':None}
        requirements = ['cheml', 'pandas', 'rdkit']
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def fit(self):
        # step1: check inputs
        # step2: assign inputs to parameters if necessary (param = @token)
        self.paramFROMinput()
        if 'molfile' in self.parameters:
            molfile = self.parameters['molfile']
            self.parameters.pop('molfile')
        elif not isinstance(self.legal_inputs['molfile'],type(None)):
            molfile = self.legal_inputs['molfile'][0]
        else:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.SuperFunction) + 'No molecule file (molfile) has been passed'
            raise TypeError(msg)
        # step3: check the dimension of input data frame
        # step4: extract parameters
        if 'path' in self.parameters:
            path = self.parameters.pop('path')
        else:
            path = None
        if 'arguments' in self.parameters:
            arguments = self.parameters.pop('arguments')
        else:
            arguments = []

        # step5: import module and make APIs
        try:
            from cheml.chem import RDKFingerprint
            model = RDKFingerprint(**self.parameters)
            model.MolfromFile(molfile,path,*arguments)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.SuperFunction) + type(err).__name__ + ': ' + err.message
            raise TypeError(msg)

        # step6: send out
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token == 'df':
                val = model.Fingerprint()
                self.Base.send[(self.iblock, token)] = [val, order.count(token)]
            elif token == 'removed_rows':
                self.Base.send[(self.iblock, token)] = [model.removed_rows, order.count(token)]
            else:
                msg = "@Task #%i(%s): asked to send a non valid output token '%s'" % (
                self.iblock + 1, self.SuperFunction, token)
                raise NameError(msg)

        # step7: delete all inputs from memory
        del self.legal_inputs

class Dragon(BASE,LIBRARY):
    def legal_IO(self):
        self.legal_inputs = {'molfile': None}
        self.legal_outputs = {'df': None}
        requirements = ['cheml', 'pandas', 'lxml', 'Dragon']
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def fit(self):
        # step1: check inputs
        # step2: assign inputs to parameters if necessary (param = @token)
        self.paramFROMinput()
        # if ('molFile' not in self.parameters or self.parameters['molFile'] is None):
        #     self.parameters['molFile'] = self.type_check('molfile', cheml_type='descriptor', req=True, py_type=str)
        # else:
        #     msg = '@Task #%i(%s): ' % (self.iblock + 1, self.SuperFunction) + 'No molecule file (molfile) has been passed'
        #     raise TypeError(msg)

        # step3: check the dimension of input data frame
        # step4: extract  parameters
        if 'script' in self.parameters:
            script = self.parameters.pop('script')
        else:
            script = 'new'
        if 'output_directory' in self.parameters:
            output_directory = self.parameters.pop('output_directory')
        else:
            output_directory = ''
        output_directory = self.Base.output_directory + '/' + output_directory

        # step5: import module and make APIs
        try:
            from cheml.chem import Dragon
            model = Dragon(**self.parameters)
            model.script_wizard(script, output_directory)
            model.run()
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.SuperFunction) + type(err).__name__ + ': ' + err.message
            raise TypeError(msg)

        # step6: send out
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

        # step7: delete dragon descriptors and all inputs from memory
        os.remove(model.output_directory + self.parameters['SaveFilePath'])
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

class MissingValues(BASE, LIBRARY):
    def legal_IO(self):
        self.legal_inputs = {'dfx': None, 'dfy':None}
        self.legal_outputs = {'dfx': None, 'dfy': None, 'api':None}
        requirements = ['cheml',  'pandas']
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def fit(self):
        # step1: check inputs
        dfx, dfx_info = self.input_check('dfx', req=True, py_type=pd.DataFrame)

        # step2: assign inputs to parameters if necessary (param = @token)
        self.paramFROMinput()

        # step3: check the dimension of input data frame
        dfx, _ = self.data_check('dfx', dfx, ndim=2, n0=None, n1=None, format_out='df')

        # step4: import module and make APIs
        try:
            from cheml.preprocessing import missing_values
            model = missing_values(**self.parameters)
            dfx = model.fit_transform(dfx)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.SuperFunction) + type(err).__name__ + ': ' + err.message
            raise TypeError(msg)

        # step5: process
        # step6: send out
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token == 'dfx':
                self.Base.send[(self.iblock, token)] = [dfx, order.count(token)]
            elif token == 'dfy':
                dfy, dfy_info = self.input_check('dfy', req=True, py_type=pd.DataFrame)
                dfy, _ = self.data_check('dfy', dfy, ndim=1, n0=None, n1=None, format_out='df')
                dfy = model.transform(dfy)
                self.Base.send[(self.iblock, token)] = [dfy, order.count(token)]
            elif token == 'api':
                self.Base.send[(self.iblock, token)] = [model, order.count(token)]
            else:
                msg = "@Task #%i(%s): asked to send a non valid output token '%s'" % (
                self.iblock + 1, self.SuperFunction, token)
                raise NameError(msg)

        # step7: delete all inputs from memory
        del self.legal_inputs

class Trimmer(BASE):
    def legal_IO(self):
        self.legal_inputs = {'dfx': None, 'dfy': None}
        self.legal_outputs = {'dfx': None, 'dfy': None, 'api': None}
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
        self.legal_outputs = {'dfx': None, 'dfy': None, 'api': None}
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

class Constant(BASE, LIBRARY):
    def legal_IO(self):
        self.legal_inputs = {'df': None}
        self.legal_outputs = {'df': None, 'api': None, 'removed_columns_': None}
        requirements = ['cheml', 'pandas']
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def fit(self):
        # step1: check inputs
        df, df_info = self.input_check('df', req=True, py_type=pd.DataFrame)

        # step2: assign inputs to parameters if necessary (param = @token)
        self.paramFROMinput()

        # step3: check the dimension of input data frame
        df, _ = self.data_check('df', df, ndim=2, n0=None, n1=None, format_out='df')

        # step4: import module and make APIs
        try:
            from cheml.preprocessing import Constant
            model = Constant()
            df = model.fit_transform(df)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.SuperFunction) + type(
                err).__name__ + ': ' + err.message
            raise TypeError(msg)

        #step5: process
        #step6: send out
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

        #step7: delete all inputs from memory
        del self.legal_inputs

#####################################################################Input

class ReadTable(BASE, LIBRARY):
    def legal_IO(self):
        self.legal_inputs = {}
        self.legal_outputs = {'df':None}
        requirements = ['cheml','pandas']
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def fit(self):
        # step1: check inputs
        # step2: assign inputs to parameters if necessary (param = @token)
        # step3: import module and make APIs
        try:
            from cheml.initialization import ReadTable
            df = ReadTable(**self.parameters)
        except Exception as err:
            msg = '@Task #%i(%s): '%(self.iblock+1, self.SuperFunction) + type(err).__name__ + ': '+ err.message
            raise TypeError(msg)

        # step4: check the dimension of input data frame
        # step5: process
        # step6: send out
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token == 'df':
                self.Base.send[(self.iblock, token)] = [df, order.count(token)]
            else:
                msg = "@Task #%i(%s): asked to send a non valid output token '%s'" % (
                    self.iblock+1,self.SuperFunction,token)
                raise NameError(msg)

        # step7: delete all inputs from memory
        del self.legal_inputs

class Merge(BASE, LIBRARY):
    def legal_IO(self):
        self.legal_inputs = {'df1':None, 'df2':None}
        self.legal_outputs = {'df':None}
        requirements = ['cheml','pandas']
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def fit(self):
        # step1: check inputs
        df1, df1_info = self.input_check('df1', req=True, py_type=pd.DataFrame)
        df2, df2_info = self.input_check('df2', req=True, py_type=pd.DataFrame)

        # step2: assign inputs to parameters if necessary (param = @token)
        self.paramFROMinput()

        # step3: check the dimension of input data frame
        df1, _ = self.data_check('df1', df1, ndim=2, n0=None, n1=None, format_out='df')
        df2, _ = self.data_check('df2', df2, ndim=2, n0=df1.shape[0], n1=None, format_out='df')

        # step4: import module and make APIs
        try:
            from cheml.initialization import Merge
            df = Merge(df1, df2)
        except Exception as err:
            msg = '@Task #%i(%s): '%(self.iblock+1, self.SuperFunction) + type(err).__name__ + ': '+ err.message
            raise TypeError(msg)

        # step5: process
        # step6: send out
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token == 'df':
                self.Base.send[(self.iblock, token)] = [df, order.count(token)]
            else:
                msg = "@Task #%i(%s): asked to send a non valid output token '%s'" % (self.iblock+1,self.SuperFunction,token)
                raise NameError(msg)

        # step7: delete all inputs from memory
        del self.legal_inputs

class Split(BASE, LIBRARY):
    def legal_IO(self):
        self.legal_inputs = {'df':None}
        self.legal_outputs = {'df1':None, 'df2':None}
        requirements = ['cheml','pandas']
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def fit(self):
        # step1: check inputs
        df, df_info = self.input_check('df', req = True, py_type = pd.DataFrame)

        # step2: assign inputs to parameters if necessary (param = @token)
        self.paramFROMinput()

        # step3: check the dimension of input data frame
        dfx, _ = self.data_check('df', df, ndim=2, n0=None, n1=None, format_out='df')

        # step4: import module and make APIs
        try:
            from cheml.initialization import Split
            split = Split(**self.parameters)
            df1, df2 = split.fit(dfx)
        except Exception as err:
            msg = '@Task #%i(%s): '%(self.iblock+1, self.SuperFunction) + type(err).__name__ + ': '+ err.message
            raise TypeError(msg)

        # step5: process
        # step6: send out
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
        # step7: delete all inputs from memory
        del self.legal_inputs

#####################################################################Output

class SaveFile(BASE, LIBRARY):
    def legal_IO(self):
        self.legal_inputs = {'df': None}
        self.legal_outputs = {'filepath': None}
        requirements = ['cheml','pandas']
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def fit(self):
        # step1: check inputs
        df, df_info = self.input_check('df', req=True, py_type=pd.DataFrame)

        # step2: assign inputs to parameters if necessary (param = @token)
        self.paramFROMinput()

        # step3: check the dimension of input data frame
        # step4: import module and make APIs
        try:
            from cheml.initialization import SaveFile
            model = SaveFile(**self.parameters)
            model.fit(self.legal_inputs['df'][0], self.Base.output_directory)
        except Exception as err:
            msg = '@Task #%i(%s): '%(self.iblock+1, self.SuperFunction) + type(err).__name__ + ': '+ err.message
            raise TypeError(msg)

        # step5: process
        # step6: send out
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token == 'filepath':
                val = model.file_path
                self.Base.send[(self.iblock, token)] = [val, order.count(token)]
            else:
                msg = "@Task #%i(%s): asked to send a non valid output token '%s'" % (self.iblock+1,self.SuperFunction,token)
                raise NameError(msg)
        # step7: delete all inputs from memory
        del self.legal_inputs

#####################################################################Regression

class NN_PSGD(BASE, LIBRARY):
    def legal_IO(self):
        self.legal_inputs = {'dfx_train': None, 'dfy_train': None, 'dfx_test':None}
        self.legal_outputs = {'dfy_train_pred': None, 'dfy_test_pred': None, 'model': None}
        requirements = ['cheml', 'pandas']
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def fit(self):
        # step1: check inputs
        dfx_train, dfx_train_info = self.input_check('dfx_train', req=True, py_type=pd.DataFrame)
        dfy_train, dfy_train_info = self.input_check('dfy_train', req=True, py_type=pd.DataFrame)

        # step2: assign inputs to parameters if necessary (param = @token)
        self.paramFROMinput()

        # step3: check the dimension of input data frame
        dfx_train, _ = self.data_check('dfx_train', dfx_train, ndim=2, n0=None, n1=None, format_out='ar')
        dfy_train, _ = self.data_check('dfy_train', dfy_train, ndim=2, n0=dfx_train.shape[0], n1=None, format_out='ar')

        # step4: import module and make APIs
        try:
            from cheml.nn import nn_psgd
            model = nn_psgd.train(dfx_train,dfy_train,**self.parameters)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.SuperFunction) + type(
                    err).__name__ + ': ' + err.message
            raise TypeError(msg)

        # step5: process
        # step6: send out
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token == 'model':
                self.Base.send[(self.iblock, token)] = [model, order.count(token)]
            elif token == 'dfy_train_pred':
                dfy_train_pred = nn_psgd.output(dfx_train, model)
                dfy_train_pred = pd.DataFrame(dfy_train_pred)  # , columns=dfy_header)
                self.Base.send[(self.iblock, token)] = [dfy_train_pred, order.count(token)]
            elif token == 'dfy_test_pred':
                dfx_test, dfx_test_info = self.input_check('dfx_test', req=True, py_type=pd.DataFrame)
                dfy_test_pred = nn_psgd.output(dfx_test, model)
                dfy_test_pred = pd.DataFrame(dfy_test_pred)  # , columns=dfy_header)
                self.Base.send[(self.iblock, token)] = [dfy_test_pred, order.count(token)]
            else:
                msg = "@Task #%i(%s): non valid output token '%s'" % (self.iblock + 1, self.SuperFunction, token)
                raise NameError(msg)

        # step7: delete all inputs from memory
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

