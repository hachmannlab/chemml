import pandas as pd
import numpy as np
import os
import warnings

from ..base import BASE

##################################################################### 2 Prepare Data

# Enter

class XYZreader(BASE):
    def fit(self):
        try:
            from cheml.initialization import XYZreader
            model = XYZreader(**self.parameters)
            molecules = model.read()
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(err).__name__ + ': ' + err.message
            raise TypeError(msg)
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token not in self.outputs:
                msg = "@Task #%i(%s): not a valid output token '%s'" % (self.iblock + 1, self.Task, token)
                raise NameError(msg)
            elif token == 'molecules':
                self.set_value(token, molecules)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]
            # elif token == 'max_n_atoms':
            #     self.set_value(token, model.max_n_atoms)
            #     self.outputs[token].count = order.count(token)
            #     self.Base.send[(self.iblock, token)] = self.outputs[token]

        del self.inputs

class load_cep_homo(BASE):
    def fit(self):
        try:
            from cheml.datasets import load_cep_homo
            smiles,homo = load_cep_homo()
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(err).__name__ + ': ' + err.message
            raise TypeError(msg)
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token not in self.outputs:
                msg = "@Task #%i(%s): not a valid output token '%s'" % (self.iblock + 1, self.Task, token)
                raise NameError(msg)
            elif token == 'smiles':
                self.set_value(token, smiles)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]
            elif token == 'homo':
                self.set_value(token, homo)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]
        del self.inputs

class load_organic_density(BASE):
    def fit(self):
        try:
            from cheml.datasets import load_organic_density
            smiles,density,features = load_organic_density()
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(err).__name__ + ': ' + err.message
            raise TypeError(msg)
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token not in self.outputs:
                msg = "@Task #%i(%s): not a valid output token '%s'" % (self.iblock + 1, self.Task, token)
                raise NameError(msg)
            elif token == 'smiles':
                self.set_value(token, smiles)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]
            elif token == 'density':
                self.set_value(token, density)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]
            elif token == 'features':
                self.set_value(token, features)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]
        del self.inputs

class load_xyz_polarizability(BASE):
    def fit(self):
        try:
            from cheml.datasets import load_xyz_polarizability
            coordinates, pol = load_xyz_polarizability()
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(err).__name__ + ': ' + err.message
            raise TypeError(msg)
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token not in self.outputs:
                msg = "@Task #%i(%s): not a valid output token '%s'" % (self.iblock + 1, self.Task, token)
                raise NameError(msg)
            elif token == 'smiles':
                self.set_value(token, coordinates)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]
            elif token == 'density':
                self.set_value(token, pol)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]
        del self.inputs

class ConvertFile(BASE):
    def fit(self):
        self.paramFROMinput()
        # self.required('file_path',req=True)
        # file_path=self.inputs['file_path'].value
        print 'from:', self.parameters['from_format']
        print 'to:', self.parameters['to_format']
        # if 'file_path' not in self.parameters and '@' not in file_path:
            # self.parameters['file_path']=file_path
        try:
            from cheml.initialization import ConvertFile
            model = ConvertFile(**self.parameters)
            converted_file_paths=model.convert()
        except Exception as err:
            msg='@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(err).__name__+': ' +err.message
            raise TypeError(msg)
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token not in self.outputs:
                msg="@Task #%i(%s): not a valid output token '%s'" % (self.iblock + 1,self.Task,token)
                raise NameError(msg)
            elif token == 'converted_file_paths':
                self.set_value(token,converted_file_paths)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock,token)] = self.outputs[token]
        del self.inputs

class PyScript(BASE):
    def fit(self):
        # step1: check inputs
        inputs = [token for token in self.inputs if self.inputs[token].value is not None]
        for token in inputs:
            code = compile("%s = self.inputs['%s'].value"%(token,token), "<string>", "exec")
            exec code
        for line in sorted(self.parameters.keys()):
            code = compile(self.parameters[line], "<string>", "exec")
            exec code
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token == 'ov1':
                self.set_value(token, ov1)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]
            elif token == 'ov2':
                self.set_value(token, ov2)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]
            elif token == 'ov3':
                self.set_value(token, ov3)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]
            elif token == 'ov4':
                self.set_value(token, ov4)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]
            elif token == 'ov5':
                self.set_value(token, ov5)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]
            elif token == 'ov6':
                self.set_value(token, ov6)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]
            else:
                msg = "@Task #%i(%s): asked to send a non valid output token '%s'" % (
                    self.iblock+1,self.Task,token)
                raise NameError(msg)
        del self.inputs


# Feature Representation

class RDKitFingerprint(BASE):
    def fit(self):
        # step1: check inputs
        # step2: assign inputs to parameters if necessary (param = @token)
        self.paramFROMinput()
        if 'molfile' in self.parameters:
            molfile = self.parameters['molfile']
            self.parameters.pop('molfile')
        else:
            self.required('molfile', req=True)
            molfile = self.inputs['molfile'].value
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
            from cheml.chem import RDKitFingerprint
            model = RDKitFingerprint(**self.parameters)
            model.MolfromFile(molfile,path,*arguments)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(err).__name__ + ': ' + err.message
            raise TypeError(msg)

        # step6: send out
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token not in self.outputs:
                msg = "@Task #%i(%s): not a valid output token '%s'" % (self.iblock + 1, self.Task, token)
                raise NameError(msg)
            elif token == 'df':
                self.set_value(token, model.Fingerprint())
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]
            elif token == 'removed_rows':
                self.set_value(token, model.removed_rows)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]

        # step7: delete all inputs from memory
        del self.inputs

class Dragon(BASE):
    def fit(self):
        # step1: check inputs
        # step2: assign inputs to parameters if necessary (param = @token)
        self.paramFROMinput()
        print 'molfile:', self.parameters['molFile']

        # step3: check the dimension of input data frame
        # step4: extract  parameters
        if 'script' in self.parameters:
            script = self.parameters.pop('script')
        else:
            script = 'new'
        # if 'output_directory' in self.parameters:
        #     output_directory = self.parameters.pop('output_directory')
        # else:
        #     output_directory = ''
        output_directory = self.Base.output_directory #+ '/' + output_directory
        if 'SaveFilePath' not in self.parameters:
            self.parameters['SaveFilePath'] = "Dragon_descriptors.txt"

        # step5: import module and make APIs
        try:
            from cheml.chem import Dragon
            model = Dragon(**self.parameters)
            model.script_wizard(script, output_directory)
            model.run()
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(err).__name__ + ': ' + err.message
            raise TypeError(msg)

        # step6: send out
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token not in self.outputs:
                msg = "@Task #%i(%s): not a valid output token '%s'" % (self.iblock + 1, self.Task, token)
                raise NameError(msg)
            elif token == 'df':
                df_path = model.data_path
                df = pd.read_csv(df_path, sep=None, engine='python')
                df = df.drop(['No.','NAME'],axis=1)
                self.set_value(token, df)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]

        # step7: delete dragon descriptors and all inputs from memory
        os.remove(model.output_directory + self.parameters['SaveFilePath'])
        del self.inputs

class Coulomb_Matrix(BASE):
    def fit(self):
        self.paramFROMinput()
        self.required('molecules', req=True)
        molecules = self.inputs['molecules'].value
        try:
            from cheml.chem import Coulomb_Matrix
            model = Coulomb_Matrix(**self.parameters)
            df = model.represent(molecules)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(err).__name__ + ': ' + err.message
            raise TypeError(msg)
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token not in self.outputs:
                msg = "@Task #%i(%s): not a valid output token '%s'" % (self.iblock + 1, self.Task, token)
                raise NameError(msg)
            elif token == 'df':
                self.set_value(token, df)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]

        del self.inputs

class Bag_of_Bonds(BASE):
    def fit(self):
        self.paramFROMinput()
        self.required('molecules', req=True)
        molecules = self.inputs['molecules'].value
        try:
            from cheml.chem import Bag_of_Bonds
            model = Bag_of_Bonds(**self.parameters)
            df = model.represent(molecules)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(err).__name__ + ': ' + err.message
            raise TypeError(msg)
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token not in self.outputs:
                msg = "@Task #%i(%s): not a valid output token '%s'" % (self.iblock + 1, self.Task, token)
                raise NameError(msg)
            elif token == 'df':
                self.set_value(token, df)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]
            # elif token == 'headers':
            #     self.set_value(token, model.headers)
            #     self.outputs[token].count = order.count(token)
            #     self.Base.send[(self.iblock, token)] = self.outputs[token]

        del self.inputs

class DistanceMatrix(BASE):
    def fit(self):
        self.paramFROMinput()
        # check inputs
        self.required('df', req=True)
        try:
            from cheml.chem import DistanceMatrix
            model = DistanceMatrix(**self.parameters)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(err).__name__ + ': ' + err.message
            raise TypeError(msg)
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token not in self.outputs:
                msg = "@Task #%i(%s): not a valid output token '%s'" % (self.iblock + 1, self.Task, token)
                raise NameError(msg)
            elif token == 'df':
                ar = self.inputs['df'].value
                df = pd.DataFrame(model.transform(ar))
                self.set_value(token, df)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]
        del self.inputs


# Preprocessor

class MissingValues(BASE):
    def fit(self):
        # parameters
        self.paramFROMinput()
        method = self.parameters.pop('func_method')
        # get df value only in case method is None, but output df is requested
        df = self.inputs['df'].value

        # process
        try:
            from cheml.preprocessing import missing_values
            if method is None:
                model = missing_values(**self.parameters)
            elif method == 'fit_transform':
                model = missing_values(**self.parameters)
                self.required('df', req=True)
                df = self.inputs['df'].value
                df, _ = self.data_check('df', df, ndim=2, n0=None, n1=None, format_out='df')
                df = model.fit_transform(df)
            elif method == 'transform':
                self.required('df', req=True)
                df = self.inputs['df'].value
                df, _ = self.data_check('df', df, ndim=2, n0=None, n1=None, format_out='df')
                self.required('api', req=True)
                model = self.inputs['api'].value
                df = model.transform(df)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(err).__name__ + ': ' + err.message
            raise TypeError(msg)

        # step6: send out
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token not in self.outputs:
                msg = "@Task #%i(%s): not a valid output token '%s'" % (self.iblock + 1, self.Task, token)
                raise NameError(msg)
            elif token == 'df':
                self.set_value(token, df)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]
            elif token == 'api':
                self.set_value(token, model)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]
        # step7: delete all inputs from memory
        del self.inputs

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
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(
                err).__name__ + ': ' + err.message
            raise TypeError(msg)
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token == 'dfx':
                self.Base.send[(self.iblock, token)] = [dfx, order.count(token),
                                                        (self.iblock, token, self.Host, self.Function)]
            elif token == 'dfy':
                self.Base.send[(self.iblock, token)] = [dfy, order.count(token),
                                                        (self.iblock, token, self.Host, self.Function)]
            elif token == 'api':
                self.Base.send[(self.iblock, token)] = [model, order.count(token),
                                                        (self.iblock, token, self.Host, self.Function)]
            else:
                msg = "@Task #%i(%s): asked to send a non valid output token '%s'" % (
                    self.iblock + 1, self.Task, token)
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
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(
                err).__name__ + ': ' + err.message
            raise TypeError(msg)
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token == 'dfx':
                self.Base.send[(self.iblock, token)] = [dfx, order.count(token),
                                                        (self.iblock, token, self.Host, self.Function)]
            elif token == 'dfy':
                self.Base.send[(self.iblock, token)] = [dfy, order.count(token),
                                                        (self.iblock, token, self.Host, self.Function)]
            elif token == 'api':
                self.Base.send[(self.iblock, token)] = [model, order.count(token),
                                                        (self.iblock, token, self.Host, self.Function)]
            else:
                msg = "@Task #%i(%s): asked to send a non valid output token '%s'" % (
                    self.iblock + 1, self.Task, token)
                raise NameError(msg)
        del self.legal_inputs



# data manipulation

class Split(BASE):
    def fit(self):
        # step1: check inputs
        self.required('df', req=True)
        df = self.inputs['df'].value

        # step2: assign inputs to parameters if necessary (param = @token)
        self.paramFROMinput()

        # step3: check the dimension of input data frame
        df, _ = self.data_check('df', df, ndim=2, n0=None, n1=None, format_out='df')

        # step4: import module and make APIs
        try:
            from cheml.initialization import Split
            split = Split(**self.parameters)
            df1, df2 = split.fit(df)
        except Exception as err:
            msg = '@Task #%i(%s): '%(self.iblock+1, self.Task) + type(err).__name__ + ': '+ err.message
            raise TypeError(msg)

        # step5: process
        # step6: send out
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token not in self.outputs:
                msg = "@Task #%i(%s): not a valid output token '%s'" % (self.iblock + 1, self.Task, token)
                raise NameError(msg)
            elif token == 'df1':
                self.set_value(token, df1)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]
            elif token == 'df2':
                self.set_value(token, df2)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]

        # step7: delete all inputs from memory
        del self.inputs

class Constant(BASE):
    def fit(self):
        # parameters
        self.paramFROMinput()
        method = self.parameters.pop('func_method')
        # get df value only in case method is None, but output df is requested
        df = self.inputs['df'].value

        # step4: import module and make APIs
        try:
            from cheml.preprocessing import Constant
            if method is None:
                model = Constant()
            elif method == 'fit_transform':
                model = Constant()
                self.required('df', req=True)
                df = self.inputs['df'].value
                df, _ = self.data_check('df', df, ndim=2, n0=None, n1=None, format_out='df')
                df = model.fit_transform(df)
            elif method == 'transform':
                self.required('df', req=True)
                df = self.inputs['df'].value
                df, _ = self.data_check('df', df, ndim=2, n0=None, n1=None, format_out='df')
                self.required('api', req=True)
                model = self.inputs['api'].value
                df = model.transform(df)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(
                err).__name__ + ': ' + err.message
            raise TypeError(msg)

        # step5: process
        # step6: send out
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token not in self.outputs:
                msg = "@Task #%i(%s): not a valid output token '%s'" % (self.iblock + 1, self.Task, token)
                raise NameError(msg)
            elif token == 'df':
                self.set_value(token, df)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]
            elif token == 'removed_columns_':
                removed_columns_ = pd.DataFrame(model.removed_columns_)
                self.set_value(token, removed_columns_)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]
            elif token == 'api':
                self.set_value(token, model)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]

        # step7: delete all inputs from memory
        del self.inputs


##################################################################### 3 Define Model

# Regression

class mlp_hogwild(BASE):
    def fit(self):
        # step1: check inputs
        self.required('dfx_train', req=True)
        dfx_train = self.inputs['dfx_train'].value
        self.required('dfy_train', req=True)
        dfy_train = self.inputs['dfy_train'].value

        # step2: assign inputs to parameters if necessary (param = @token)
        self.paramFROMinput()
        method = self.parameters.pop('func_method')

        # step3: check the dimension of input data frame
        dfx_train, _ = self.data_check('dfx_train', dfx_train, ndim=2, n0=None, n1=None, format_out='ar')
        dfy_train, _ = self.data_check('dfy_train', dfy_train, ndim=2, n0=dfx_train.shape[0], n1=None, format_out='ar')

        # step4: import module and make APIs
        try:
            from cheml.nn import mlp_hogwild
            if method is None:
                model = mlp_hogwild(**self.parameters)
            elif method == 'fit':
                model = mlp_hogwild(**self.parameters)
                self.required('dfx', req=True)
                dfx = self.inputs['dfx'].value
                dfx, _ = self.data_check('dfx', dfx, ndim=2, n0=None, n1=None, format_out='ar')
                self.required('dfy', req=True)
                dfy = self.inputs['dfy'].value
                dfy, _ = self.data_check('dfy', dfy, ndim=2, n0=dfx.shape[0], n1=None, format_out='ar')
                model.fit(dfx,dfy)
            elif method == 'predict':
                self.required('dfx', req=True)
                self.required('api', req=True)
                dfx = self.inputs['dfx'].value
                dfx, _ = self.data_check('dfx', dfx, ndim=2, n0=None, n1=None, format_out='ar')
                api = self.inputs['api'].value
                dfy_predict = api.predict(dfx)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(
                    err).__name__ + ': ' + err.message
            raise TypeError(msg)

        # step5: process
        # step6: send out
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token not in self.outputs:
                msg = "@Task #%i(%s): not a valid output token '%s'" % (self.iblock + 1, self.Task, token)
                raise NameError(msg)
            elif token == 'api':
                self.set_value(token, model)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]
            elif token == 'dfy_predict':
                self.set_value(token, dfy_predict)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]

        # step7: delete all inputs from memory
        del self.inputs

class mlp_dsgd(BASE):
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
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(
                err).__name__ + ': ' + err.message
            raise TypeError(msg)

        dfy_pred = nn_psgd.output(dfx_train,model)
        dfy_pred = pd.DataFrame(dfy_pred, columns=dfy_header)

        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token == 'model':
                self.Base.send[(self.iblock, token)] = [model, order.count(token),
                                                        (self.iblock, token, self.Host, self.Function)]
            elif token == 'dfy_train_pred':
                self.Base.send[(self.iblock, token)] = [dfy_pred, order.count(token),
                                                        (self.iblock, token, self.Host, self.Function)]
            else:
                msg = "@Task #%i(%s): non valid output token '%s'" % (self.iblock + 1, self.Task, token)
                raise NameError(msg)
        del self.legal_inputs

##################################################################### 6 Mix

class kfold_pool(BASE):
    def legal_IO(self):
        self.legal_inputs = {'dfx': None, 'dfy': None, 'kfold':None, 'model':None, 'evaluator':None}
        self.legal_outputs = {'evaluation_results_': None, 'best_model_': None}
        requirements = ['scikit_learn']
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def fit(self):
        # step2: assign inputs to parameters if necessary (param = @token)
        self.paramFROMinput()

        # step4: import module and make APIs
        try:
            self._reg_evaluation_params()
        except Exception as err:
            msg = '@Task #%i(%s): '%(self.iblock+1, self.Task) + type(err).__name__ + ': '+ err.message
            raise TypeError(msg)

        # step5: process
        # step6: send out
        order = [edge[1] for edge in self.Base.graph if edge[0]==self.iblock]
        for token in set(order):
            if token == 'evaluator':
                self.Base.send[(self.iblock, token)] = [self.evaluator, order.count(token),
                                                        (self.iblock, token, self.Host, self.Function)]
            elif token == 'evaluation_results_':
                # step1: check inputs
                dfy, dfy_info = self.input_check('dfy', req=True, py_type=pd.DataFrame)
                dfy_pred, dfy_pred_info = self.input_check('dfy_pred', req=True, py_type=pd.DataFrame)

                # step3: check the dimension of input data frame
                dfy, _ = self.data_check('dfy', dfy, ndim=2, n0=None, n1=None, format_out='df')
                dfy_pred, _ = self.data_check('dfy_pred', dfy_pred, ndim=2, n0=dfy.shape[0], n1=None, format_out='df')

                self._reg_evaluate(dfy, dfy_pred, self.evaluator)
                evaluation_results_ = self.results
                self.Base.send[(self.iblock, token)] = [pd.DataFrame(evaluation_results_), order.count(token),
                                                        (self.iblock,token,self.Host,self.Function)]
            else:
                msg = "@Task #%i(%s): non valid output token '%s'" % (self.iblock+1, self.Task, token)
                raise NameError(msg)

        #step7: delete all inputs from memory
        del self.legal_inputs

##################################################################### 7 Store

class SaveFile(BASE):
    def fit(self):
        # step1: check inputs
        self.required('df', req=True)
        df = self.inputs['df'].value

        # step2: assign inputs to parameters if necessary (param = @token)
        self.paramFROMinput()

        # step3: check the dimension of input data frame
        # step4: import module and make APIs
        try:
            from cheml.initialization import SaveFile
            model = SaveFile(**self.parameters)
            model.fit(df, self.Base.output_directory)
        except Exception as err:
            msg = '@Task #%i(%s): '%(self.iblock+1, self.Task) + type(err).__name__ + ': '+ err.message
            raise TypeError(msg)

        # step5: process
        # step6: send out
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token not in self.outputs:
                msg = "@Task #%i(%s): not a valid output token '%s'" % (self.iblock + 1, self.Task, token)
                raise NameError(msg)
            elif token == 'filepath':
                self.set_value(token, model.file_path)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]

        # step7: delete all inputs from memory
        del self.inputs


